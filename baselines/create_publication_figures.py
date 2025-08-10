#!/usr/bin/env python3
"""
Create clean, publication-ready figures for opponent diversity research.
No annotations or claims - just clean results with proper diversity tags.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_tournament_data():
    """Load and preprocess tournament results."""
    df = pd.read_csv('tournament_results/tournament_results_20250802_183321.csv')
    return df

def create_performance_comparison_figure(df):
    """Figure 1: Overall Performance Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['SPPPO\n(No Opponent Diversity)', 'IPPO\n(Minimal Opponent Diversity)', 'FSPPPO\n(Historical Opponent Diversity)']
    algorithm_names = ['SPPPO_latest', 'IPPO_latest_', 'FSPPPO_latest']
    win_rates = []
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, orange, green
    
    for algo_name in algorithm_names:
        algo_matches = df[(df['player1'] == algo_name) | (df['player2'] == algo_name)]
        wins = len(algo_matches[algo_matches['winner'] == algo_name])
        total = len(algo_matches)
        win_rate = wins / total * 100
        win_rates.append(win_rate)
    
    bars = ax.bar(algorithms, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 75)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig('tournament_results/figure1_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tournament_results/figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_generalization_heatmap(df):
    """Figure 2: Generalization to Scripted Opponents"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Y-axis: More diversity at top, no diversity at bottom
    algorithms = ['FSPPPO\n(Historical Opponent Diversity)', 'IPPO\n(Minimal Opponent Diversity)', 'SPPPO\n(No Opponent Diversity)']
    algorithm_names = ['FSPPPO_latest', 'IPPO_latest_', 'SPPPO_latest']
    # X-axis: Best FSPPPO performance left, worst performance right (Random, Noop, Dodge, Seek, Guardian)
    scripted_opponents = ['scripted_random', 'scripted_noop', 'scripted_dodge', 'scripted_seek', 'scripted_guardian']
    
    # Create matrix of win rates
    win_matrix = []
    for algo_name in algorithm_names:
        algo_wins = []
        for opponent in scripted_opponents:
            matches = df[((df['player1'] == algo_name) & (df['player2'] == opponent)) | 
                        ((df['player1'] == opponent) & (df['player2'] == algo_name))]
            wins = len(matches[matches['winner'] == algo_name])
            total = len(matches)
            win_rate = wins / total * 100 if total > 0 else 0
            algo_wins.append(win_rate)
        win_matrix.append(algo_wins)
    
    # Create heatmap with better aspect ratio
    win_matrix = np.array(win_matrix)
    im = ax.imshow(win_matrix, cmap='RdYlGn', aspect='equal', vmin=0, vmax=100)
    
    # Set ticks and labels with better spacing
    ax.set_xticks(range(len(scripted_opponents)))
    ax.set_yticks(range(len(algorithms)))
    ax.set_xticklabels([opp.replace('scripted_', '').title() for opp in scripted_opponents], fontsize=14, fontweight='bold')
    ax.set_yticklabels(algorithms, fontsize=12, fontweight='bold')
    
    # Add text annotations with better contrast and size
    for i in range(len(algorithms)):
        for j in range(len(scripted_opponents)):
            win_rate = win_matrix[i, j]
            # Choose text color based on background for better contrast
            text_color = 'white' if win_rate < 50 else 'black'
            text = ax.text(j, i, f'{win_rate:.0f}%',
                          ha="center", va="center", color=text_color, 
                          fontweight='bold', fontsize=16)
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Win Rate (%)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    ax.set_xlabel('Scripted Opponents', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Learning Algorithm', fontsize=16, fontweight='bold', labelpad=10)
    
    # Remove grid lines for cleaner appearance
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('tournament_results/figure2_generalization_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tournament_results/figure2_generalization_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_outcome_distribution_figure(df):
    """Figure 3: Win/Draw/Loss Distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = ['SPPPO\n(No Opponent Diversity)', 'IPPO\n(Minimal Opponent Diversity)', 'FSPPPO\n(Historical Opponent Diversity)']
    algorithm_names = ['SPPPO_latest', 'IPPO_latest_', 'FSPPPO_latest']
    
    win_rates = []
    draw_rates = []
    loss_rates = []
    
    for algo_name in algorithm_names:
        algo_matches = df[(df['player1'] == algo_name) | (df['player2'] == algo_name)]
        wins = len(algo_matches[algo_matches['winner'] == algo_name])
        draws = len(algo_matches[algo_matches['winner'] == 'draw'])
        losses = len(algo_matches) - wins - draws
        total = len(algo_matches)
        
        win_rates.append(wins / total * 100)
        draw_rates.append(draws / total * 100)
        loss_rates.append(losses / total * 100)
    
    # Stacked bar chart
    width = 0.6
    x = np.arange(len(algorithms))
    
    p1 = ax.bar(x, win_rates, width, label='Wins', color='#27ae60', alpha=0.8)
    p2 = ax.bar(x, draw_rates, width, bottom=win_rates, label='Draws', color='#f39c12', alpha=0.8)
    p3 = ax.bar(x, loss_rates, width, bottom=np.array(win_rates) + np.array(draw_rates), 
                label='Losses', color='#e74c3c', alpha=0.8)
    
    # Add percentage labels
    for i, (w, d, l) in enumerate(zip(win_rates, draw_rates, loss_rates)):
        ax.text(i, w/2, f'{w:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        ax.text(i, w + d/2, f'{d:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        if l > 5:  # Only show loss percentage if significant
            ax.text(i, w + d + l/2, f'{l:.1f}%', ha='center', va='center', fontweight='bold', color='white')
    
    ax.set_ylabel('Percentage of Episodes (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('tournament_results/figure3_outcome_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tournament_results/figure3_outcome_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_robustness_radar_figure(df):
    """Figure 4: Performance Across Opponent Types"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    algorithms = ['IPPO (Minimal)', 'SPPPO (None)', 'FSPPPO (Historical)']
    algorithm_names = ['IPPO_latest_', 'SPPPO_latest', 'FSPPPO_latest']
    scripted_opponents = ['scripted_noop', 'scripted_random', 'scripted_seek', 'scripted_guardian', 'scripted_dodge']
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(scripted_opponents), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#f39c12', '#e74c3c', '#27ae60']
    
    for i, (algo, algo_name) in enumerate(zip(algorithms, algorithm_names)):
        win_rates = []
        for opponent in scripted_opponents:
            matches = df[((df['player1'] == algo_name) & (df['player2'] == opponent)) | 
                        ((df['player1'] == opponent) & (df['player2'] == algo_name))]
            wins = len(matches[matches['winner'] == algo_name])
            total = len(matches)
            win_rate = wins / total * 100 if total > 0 else 0
            win_rates.append(win_rate)
        
        win_rates += win_rates[:1]  # Complete the circle
        
        ax.plot(angles, win_rates, 'o-', linewidth=3, label=algo, color=colors[i], markersize=8)
        ax.fill(angles, win_rates, alpha=0.25, color=colors[i])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([opp.replace('scripted_', '').title() for opp in scripted_opponents], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('tournament_results/figure4_robustness_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tournament_results/figure4_robustness_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_diversity_performance_correlation(df):
    """Figure 5: Training Diversity vs Performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points
    diversity_scores = [0, 1, 5]
    algorithms = ['SPPPO\n(No Diversity)', 'IPPO\n(Minimal Diversity)', 'FSPPPO\n(Historical Diversity)']
    algorithm_names = ['SPPPO_latest', 'IPPO_latest_', 'FSPPPO_latest']
    
    # Calculate overall win rates and generalization rates
    overall_win_rates = []
    generalization_rates = []
    
    for algo_name in algorithm_names:
        # Overall performance
        algo_matches = df[(df['player1'] == algo_name) | (df['player2'] == algo_name)]
        wins = len(algo_matches[algo_matches['winner'] == algo_name])
        total = len(algo_matches)
        overall_win_rates.append(wins / total * 100)
        
        # Generalization (vs scripted opponents)
        scripted_opponents = ['scripted_noop', 'scripted_random', 'scripted_seek', 'scripted_guardian', 'scripted_dodge']
        scripted_wins = 0
        scripted_total = 0
        for opponent in scripted_opponents:
            matches = df[((df['player1'] == algo_name) & (df['player2'] == opponent)) | 
                        ((df['player1'] == opponent) & (df['player2'] == algo_name))]
            wins = len(matches[matches['winner'] == algo_name])
            scripted_wins += wins
            scripted_total += len(matches)
        generalization_rates.append(scripted_wins / scripted_total * 100)
    
    # Create scatter plot with trend lines
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    sizes = [200, 300, 400]  # Larger for higher diversity
    
    # Plot overall performance
    scatter1 = ax.scatter(diversity_scores, overall_win_rates, c=colors, s=sizes, 
                         alpha=0.7, edgecolors='black', linewidth=2, label='Overall Performance')
    
    # Plot generalization performance
    scatter2 = ax.scatter(diversity_scores, generalization_rates, c=colors, s=sizes, 
                         alpha=0.7, edgecolors='black', linewidth=2, marker='^', 
                         label='Generalization (vs Scripted)')
    
    # Add trend lines
    z1 = np.polyfit(diversity_scores, overall_win_rates, 1)
    p1 = np.poly1d(z1)
    ax.plot(diversity_scores, p1(diversity_scores), "--", color='blue', linewidth=2, alpha=0.8)
    
    z2 = np.polyfit(diversity_scores, generalization_rates, 1)
    p2 = np.poly1d(z2)
    ax.plot(diversity_scores, p2(diversity_scores), "--", color='red', linewidth=2, alpha=0.8)
    
    # Add algorithm labels
    for i, (x, y1, y2, algo) in enumerate(zip(diversity_scores, overall_win_rates, generalization_rates, algorithms)):
        ax.annotate(algo, (x, max(y1, y2) + 5), 
                   ha='center', va='bottom', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Training Opponent Diversity Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('tournament_results/figure5_diversity_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tournament_results/figure5_diversity_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Create all publication-ready figures."""
    print("🎨 Creating Publication-Ready Figures...")
    print("=" * 50)
    
    # Load data
    df = load_tournament_data()
    
    # Create output directory
    Path('tournament_results').mkdir(exist_ok=True)
    
    # Generate all figures
    print("📊 Figure 1: Performance Comparison")
    create_performance_comparison_figure(df)
    
    print("📊 Figure 2: Generalization Heatmap")
    create_generalization_heatmap(df)
    
    print("📊 Figure 3: Outcome Distribution")
    create_outcome_distribution_figure(df)
    
    print("📊 Figure 4: Robustness Radar")
    create_robustness_radar_figure(df)
    
    print("📊 Figure 5: Diversity-Performance Correlation")
    create_diversity_performance_correlation(df)
    
    print("\n✅ All publication figures created successfully!")
    print("📁 Figures saved in: tournament_results/")
    print("\nFigure Files (PDF and PNG):")
    print("- figure1_performance_comparison")
    print("- figure2_generalization_heatmap") 
    print("- figure3_outcome_distribution")
    print("- figure4_robustness_radar")
    print("- figure5_diversity_correlation")

if __name__ == "__main__":
    main()
