#!/usr/bin/env python3
"""
Tournament Data Analysis and Visualization Pipeline

Step 2 of the two-step evaluation pipeline. Analyzes raw tournament data
and generates comprehensive visualizations to identify algorithm weaknesses.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

plt.style.use('default')
sns.set_palette("husl")


class TournamentDataAnalyzer:
    """Comprehensive tournament data analysis and visualization."""
    
    def __init__(self, data_dir: str, output_dir: str = None, output_format: str = 'png'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "analysis"
        self.output_format = output_format
        self.output_dir.mkdir(exist_ok=True)
        
        self.tournament_summary = None
        self.all_episodes_df = None
        self.match_summaries = {}
        self.player_types = {}
        
        print(f"📊 Tournament Data Analyzer initialized")
        print(f"📁 Data: {self.data_dir} → Output: {self.output_dir}")
    
    def load_data(self):
        """Load all tournament data."""
        print("\n📥 Loading tournament data...")
        
        # Find the most recent tournament results CSV
        csv_files = list(self.data_dir.glob("tournament_results_*.csv"))
        if not csv_files:
            print("❌ No tournament results CSV found")
            return False
        
        # Use the most recent CSV file
        latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"📄 Using tournament data: {latest_csv.name}")
        
        # Load episodes CSV
        try:
            self.all_episodes_df = pd.read_csv(latest_csv)
            print(f"✅ Loaded {len(self.all_episodes_df)} episodes")
        except Exception as e:
            print(f"❌ Failed to load episodes data: {e}")
            return False
        
        # Extract player types from the data
        all_players = set(self.all_episodes_df['player1'].unique()) | set(self.all_episodes_df['player2'].unique())
        for player in all_players:
            if player.startswith('scripted_'):
                self.player_types[player] = {
                    'type': 'scripted',
                    'algorithm': 'scripted'
                }
            else:
                # Extract algorithm from player name (e.g., 'IPPO_latest_' -> 'IPPO')
                algorithm = player.split('_')[0]
                self.player_types[player] = {
                    'type': 'checkpoint',
                    'algorithm': algorithm
                }
        
        print(f"✅ Identified {len(self.player_types)} players: {list(self.player_types.keys())}")
        return True
    
    def create_win_rate_matrix(self) -> pd.DataFrame:
        """Create win rate matrix for all player matchups."""
        print("\n📈 Creating win rate matrix...")
        
        # Get all unique players from the CSV data
        players = sorted(list(set(self.all_episodes_df['player1'].unique()) | 
                             set(self.all_episodes_df['player2'].unique())))
        
        win_rate_matrix = pd.DataFrame(index=players, columns=players, dtype=float)
        
        # Fill diagonal with NaN (players don't play against themselves)
        for player in players:
            win_rate_matrix.loc[player, player] = np.nan
        
        # Calculate win rates from episode data
        for p1 in players:
            for p2 in players:
                if p1 == p2:
                    continue
                    
                # Get all episodes between these two players
                matchup_episodes = self.all_episodes_df[
                    ((self.all_episodes_df['player1'] == p1) & (self.all_episodes_df['player2'] == p2)) |
                    ((self.all_episodes_df['player1'] == p2) & (self.all_episodes_df['player2'] == p1))
                ]
                
                if len(matchup_episodes) > 0:
                    # Count wins for p1
                    p1_wins = len(matchup_episodes[matchup_episodes['winner'] == p1])
                    total_games = len(matchup_episodes)
                    win_rate = p1_wins / total_games if total_games > 0 else 0.0
                    win_rate_matrix.loc[p1, p2] = win_rate
        
        return win_rate_matrix
    
    def plot_win_rate_heatmap(self, win_rate_matrix: pd.DataFrame):
        """Create win rate heatmap."""
        print("\n🎨 Creating win rate heatmap...")
        
        plt.figure(figsize=(12, 10))
        mask = win_rate_matrix.isnull()
        sns.heatmap(win_rate_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   center=0.5, vmin=0, vmax=1, mask=mask,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Win Rate Matrix - All Player Matchups', fontsize=16, fontweight='bold')
        plt.xlabel('Opponent', fontsize=12)
        plt.ylabel('Player', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_file = self.output_dir / f"win_rate_heatmap.{self.output_format}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_file}")
    
    def analyze_algorithm_performance(self):
        """Analyze performance differences between algorithms with comprehensive metrics."""
        print("\n🔬 Analyzing algorithm performance...")
        
        algorithm_performance = defaultdict(list)
        
        for player_name, player_info in self.player_types.items():
            if player_info['type'] == 'checkpoint':
                algorithm = player_info['algorithm']
                
                player_episodes = self.all_episodes_df[
                    (self.all_episodes_df['player1'] == player_name) |
                    (self.all_episodes_df['player2'] == player_name)
                ]
                
                # Calculate win/loss/draw breakdown
                wins = len(player_episodes[player_episodes['winner'] == player_name])
                losses = len(player_episodes[
                    (player_episodes['winner'] != player_name) & 
                    (player_episodes['winner'] != 'draw')
                ])
                draws = len(player_episodes[player_episodes['winner'] == 'draw'])
                total_games = len(player_episodes)
                
                win_rate = wins / total_games if total_games > 0 else 0
                loss_rate = losses / total_games if total_games > 0 else 0
                draw_rate = draws / total_games if total_games > 0 else 0
                
                # Calculate mean rewards
                player1_episodes = player_episodes[player_episodes['player1'] == player_name]
                player2_episodes = player_episodes[player_episodes['player2'] == player_name]
                
                p1_rewards = player1_episodes['player1_reward'].values if len(player1_episodes) > 0 else []
                p2_rewards = player2_episodes['player2_reward'].values if len(player2_episodes) > 0 else []
                all_rewards = list(p1_rewards) + list(p2_rewards)
                mean_reward = np.mean(all_rewards) if len(all_rewards) > 0 else 0
                
                # Performance vs scripted
                scripted_opponents = [p for p, info in self.player_types.items() 
                                    if info['type'] == 'scripted']
                scripted_episodes = player_episodes[
                    (player_episodes['player1'].isin(scripted_opponents)) |
                    (player_episodes['player2'].isin(scripted_opponents))
                ]
                
                scripted_wins = len(scripted_episodes[scripted_episodes['winner'] == player_name])
                scripted_losses = len(scripted_episodes[
                    (scripted_episodes['winner'] != player_name) & 
                    (scripted_episodes['winner'] != 'draw')
                ])
                scripted_draws = len(scripted_episodes[scripted_episodes['winner'] == 'draw'])
                scripted_total = len(scripted_episodes)
                
                scripted_win_rate = scripted_wins / scripted_total if scripted_total > 0 else 0
                scripted_loss_rate = scripted_losses / scripted_total if scripted_total > 0 else 0
                scripted_draw_rate = scripted_draws / scripted_total if scripted_total > 0 else 0
                
                # Mean reward vs scripted
                scripted_p1 = scripted_episodes[scripted_episodes['player1'] == player_name]
                scripted_p2 = scripted_episodes[scripted_episodes['player2'] == player_name]
                scripted_p1_rewards = scripted_p1['player1_reward'].values if len(scripted_p1) > 0 else []
                scripted_p2_rewards = scripted_p2['player2_reward'].values if len(scripted_p2) > 0 else []
                scripted_all_rewards = list(scripted_p1_rewards) + list(scripted_p2_rewards)
                scripted_mean_reward = np.mean(scripted_all_rewards) if len(scripted_all_rewards) > 0 else 0
                
                # Performance vs learned
                learned_opponents = [p for p, info in self.player_types.items() 
                                   if info['type'] == 'checkpoint' and p != player_name]
                learned_episodes = player_episodes[
                    (player_episodes['player1'].isin(learned_opponents)) |
                    (player_episodes['player2'].isin(learned_opponents))
                ]
                learned_wins = len(learned_episodes[learned_episodes['winner'] == player_name])
                learned_total = len(learned_episodes)
                learned_win_rate = learned_wins / learned_total if learned_total > 0 else 0
                
                algorithm_performance[algorithm].append({
                    'player': player_name,
                    'overall_win_rate': win_rate,
                    'overall_loss_rate': loss_rate,
                    'overall_draw_rate': draw_rate,
                    'overall_mean_reward': mean_reward,
                    'vs_scripted_win_rate': scripted_win_rate,
                    'vs_scripted_loss_rate': scripted_loss_rate,
                    'vs_scripted_draw_rate': scripted_draw_rate,
                    'vs_scripted_mean_reward': scripted_mean_reward,
                    'vs_learned_win_rate': learned_win_rate,
                    'generalization_gap': scripted_win_rate - learned_win_rate
                })
        
        # Create performance plots
        self.plot_algorithm_performance(algorithm_performance)
        
        # Save performance data
        performance_data = []
        for algorithm, players in algorithm_performance.items():
            for player_data in players:
                player_data['algorithm'] = algorithm
                performance_data.append(player_data)
        
        performance_df = pd.DataFrame(performance_data)
        performance_file = self.output_dir / "algorithm_performance.csv"
        performance_df.to_csv(performance_file, index=False)
        print(f"💾 Saved: {performance_file}")
        
        return performance_df
    
    def plot_algorithm_performance(self, algorithm_performance: Dict):
        """Create algorithm performance comparison plots."""
        print("\n🎨 Creating algorithm performance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(algorithm_performance.keys())
        overall_rates = [[p['overall_win_rate'] for p in algorithm_performance[alg]] for alg in algorithms]
        scripted_rates = [[p['vs_scripted_win_rate'] for p in algorithm_performance[alg]] for alg in algorithms]
        learned_rates = [[p['vs_learned_win_rate'] for p in algorithm_performance[alg]] for alg in algorithms]
        gaps = [[p['generalization_gap'] for p in algorithm_performance[alg]] for alg in algorithms]
        
        # Filter out empty data lists
        valid_algorithms = []
        valid_overall_rates = []
        valid_scripted_rates = []
        valid_learned_rates = []
        valid_gaps = []
        
        for i, alg in enumerate(algorithms):
            if overall_rates[i]:  # Only include algorithms with data
                valid_algorithms.append(alg)
                valid_overall_rates.append(overall_rates[i])
                valid_scripted_rates.append(scripted_rates[i])
                valid_learned_rates.append(learned_rates[i])
                valid_gaps.append(gaps[i])
        
        if not valid_algorithms:
            print("⚠️ No valid algorithm data found for plotting")
            return
        
        # Overall performance
        axes[0, 0].boxplot(valid_overall_rates, tick_labels=valid_algorithms)
        axes[0, 0].set_title('Overall Win Rates', fontweight='bold')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # vs Scripted
        axes[0, 1].boxplot(valid_scripted_rates, tick_labels=valid_algorithms)
        axes[0, 1].set_title('vs Scripted Opponents', fontweight='bold')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # vs Learned
        axes[1, 0].boxplot(valid_learned_rates, tick_labels=valid_algorithms)
        axes[1, 0].set_title('vs Learned Algorithms', fontweight='bold')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Generalization gap
        axes[1, 1].boxplot(valid_gaps, tick_labels=valid_algorithms)
        axes[1, 1].set_title('Generalization Gap\n(Scripted - Learned)', fontweight='bold')
        axes[1, 1].set_ylabel('Performance Gap')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"algorithm_performance.{self.output_format}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_file}")
    
    def analyze_episode_lengths(self):
        """Analyze episode length patterns."""
        print("\n📏 Analyzing episode lengths...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode length distribution
        self.all_episodes_df['episode_length'].hist(bins=20, ax=axes[0, 0], alpha=0.7)
        axes[0, 0].axvline(self.all_episodes_df['episode_length'].mean(), 
                          color='red', linestyle='--', 
                          label=f"Mean: {self.all_episodes_df['episode_length'].mean():.1f}")
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Length by outcome
        outcome_data = []
        for _, episode in self.all_episodes_df.iterrows():
            # Since we don't have termination_reason, use episode length and winner to infer outcome
            if episode['episode_length'] >= 100:  # Assuming max episode length is 100
                outcome_data.append(('Timeout', episode['episode_length']))
            elif episode['winner'] == 'draw' or pd.isna(episode['winner']):
                outcome_data.append(('Draw', episode['episode_length']))
            else:
                outcome_data.append(('Decisive', episode['episode_length']))
        
        outcome_df = pd.DataFrame(outcome_data, columns=['outcome', 'length'])
        outcome_df.boxplot(column='length', by='outcome', ax=axes[0, 1])
        axes[0, 1].set_title('Episode Length by Outcome')
        
        # Decisiveness by player type
        decisiveness_data = []
        for player_name, player_info in self.player_types.items():
            player_episodes = self.all_episodes_df[
                (self.all_episodes_df['player1'] == player_name) |
                (self.all_episodes_df['player2'] == player_name)
            ]
            if len(player_episodes) > 0:
                # Use fallback max_steps if tournament_summary is not available
                max_steps = 100  # Default based on tournament evaluation configuration
                if self.tournament_summary and 'tournament_info' in self.tournament_summary:
                    max_steps = self.tournament_summary['tournament_info']['max_episode_steps']
                decisive_rate = len(player_episodes[player_episodes['episode_length'] < max_steps]) / len(player_episodes)
                decisiveness_data.append((player_info['type'], decisive_rate))
        
        decisiveness_df = pd.DataFrame(decisiveness_data, columns=['type', 'decisiveness'])
        decisiveness_by_type = decisiveness_df.groupby('type')['decisiveness'].mean()
        decisiveness_by_type.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Decisiveness by Player Type')
        axes[1, 0].set_ylabel('Decisiveness Rate')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        # Average episode length by player
        length_data = []
        for player_name in self.player_types.keys():
            player_episodes = self.all_episodes_df[
                (self.all_episodes_df['player1'] == player_name) |
                (self.all_episodes_df['player2'] == player_name)
            ]
            if len(player_episodes) > 0:
                avg_length = player_episodes['episode_length'].mean()
                length_data.append((player_name, avg_length))
        
        length_df = pd.DataFrame(length_data, columns=['player', 'avg_length'])
        length_df.plot(x='player', y='avg_length', kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Episode Length by Player')
        axes[1, 1].set_ylabel('Average Length')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"episode_analysis.{self.output_format}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("\n🚀 Starting comprehensive analysis...")
        
        if not self.load_data():
            return
        
        # Win rate analysis
        win_rate_matrix = self.create_win_rate_matrix()
        self.plot_win_rate_heatmap(win_rate_matrix)
        
        # Algorithm performance
        performance_df = self.analyze_algorithm_performance()
        
        # Episode length analysis
        self.analyze_episode_lengths()
        
        print(f"\n🎉 Analysis complete! Results in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tournament data analysis")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing tournament data")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: data-dir/analysis)")
    parser.add_argument("--output-format", type=str, default="png",
                       choices=["png", "pdf", "svg"], help="Output format")
    
    args = parser.parse_args()
    
    analyzer = TournamentDataAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_format=args.output_format
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
