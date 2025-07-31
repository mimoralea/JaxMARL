#!/usr/bin/env python3
"""
Comprehensive Tournament Data Collection Script

Step 1 of the two-step evaluation pipeline. This script runs extensive tournaments
and collects detailed raw data for later analysis. Designed to identify weaknesses
in baseline algorithms when trained without sufficient opponent diversity.

Features:
- Comprehensive matchup matrix (all vs all, including scripted vs scripted)
- Detailed episode-level data collection
- Win/Loss/Draw tracking with episode length analysis
- Raw observations and actions logging (optional)
- Extensible data format for future metrics
- Support for multiple seeds and statistical significance

Usage:
    python -m baselines.comprehensive_tournament_data_collection
    python -m baselines.comprehensive_tournament_data_collection --include-observations
    python -m baselines.comprehensive_tournament_data_collection --episodes-per-matchup 200
"""

import argparse
import os
import sys
import json
import csv
import time
import itertools
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

# Environment imports
from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper

# Import existing tournament infrastructure
try:
    from baselines.tournament_evaluation import TournamentPlayer, TournamentEvaluator
except ImportError:
    from tournament_evaluation import TournamentPlayer, TournamentEvaluator


@dataclass
class EpisodeData:
    """Detailed data for a single episode."""
    episode_id: int
    match_id: str
    player1_name: str
    player2_name: str
    player1_position: str  # 'agent_0' or 'agent_1'
    player2_position: str
    winner: Optional[str]
    episode_length: int
    player1_total_reward: float
    player2_total_reward: float
    player1_final_reward: float
    player2_final_reward: float
    termination_reason: str  # 'max_steps', 'winner', 'draw'
    timestamp: str
    # Optional detailed data
    observations: Optional[List] = None
    actions: Optional[List] = None
    rewards_sequence: Optional[List] = None


@dataclass
class MatchData:
    """Summary data for a complete match between two players."""
    match_id: str
    player1_name: str
    player2_name: str
    total_episodes: int
    player1_wins: int
    player2_wins: int
    draws: int
    player1_win_rate: float
    player2_win_rate: float
    draw_rate: float
    avg_episode_length: float
    player1_avg_reward: float
    player2_avg_reward: float
    episodes: List[EpisodeData]


class ComprehensiveTournamentDataCollector(TournamentEvaluator):
    """Enhanced tournament evaluator with comprehensive data collection."""
    
    def __init__(self, evaluator: TournamentEvaluator, output_dir: str = "tournament_data",
                 episodes_per_matchup: int = 100, include_observations: bool = False,
                 include_actions: bool = True, include_rewards_sequence: bool = True):
        """Initialize comprehensive tournament data collector."""
        self.evaluator = evaluator  # Store evaluator reference
        self.output_dir = Path(output_dir)
        self.episodes_per_matchup = episodes_per_matchup
        self.include_observations = include_observations
        self.include_actions = include_actions
        self.include_rewards_sequence = include_rewards_sequence
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize environment (same as evaluator)
        self.env_name = evaluator.env_name
        self.env = evaluator.env
        self.max_episode_steps = evaluator.max_episode_steps
        self.episodes_per_side = episodes_per_matchup // 2
        
        # Initialize other tournament attributes
        self.players = []
        self.matches = []
        self.results = []
        
        # Enhanced data storage
        self.episode_data: List[EpisodeData] = []
        self.match_data: List[MatchData] = []
        
        # Create detailed output structure
        self.raw_data_dir = self.output_dir / "raw_data"
        self.match_summaries_dir = self.output_dir / "match_summaries"
        self.raw_data_dir.mkdir(exist_ok=True)
        self.match_summaries_dir.mkdir(exist_ok=True)
        
        print(f"🔬 Comprehensive data collection initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Episodes per matchup: {episodes_per_matchup}")
        print(f"🎯 Include observations: {include_observations}")
        print(f"🎮 Include actions: {include_actions}")
        print(f"💰 Include reward sequences: {include_rewards_sequence}")
    
    def run_single_episode_detailed(self, player1: TournamentPlayer, player2: TournamentPlayer,
                                  episode_id: int, match_id: str, rng_key) -> EpisodeData:
        """Run a single episode with detailed data collection."""
        
        # Reset environment
        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = self.env.reset(reset_key)
        
        # Initialize data collection
        observations_log = [] if self.include_observations else None
        actions_log = [] if self.include_actions else None
        rewards_log = [] if self.include_rewards_sequence else None
        
        episode_rewards = {agent: 0.0 for agent in self.env.agents}
        step_count = 0
        done = False
        
        while not done and step_count < self.max_episode_steps:
            # Get actions from both players
            actions = {}
            step_actions = {} if self.include_actions else None
            
            for i, (agent, player) in enumerate([(self.env.agents[0], player1), 
                                                (self.env.agents[1], player2)]):
                if player.player_type == 'scripted':
                    rng_key, action_key = jax.random.split(rng_key)
                    action = self.get_scripted_action(obs[agent], player.name, action_key)
                else:
                    # Get action from trained model
                    obs_array = jnp.array(obs[agent])
                    if len(obs_array.shape) == 1:
                        obs_array = obs_array[None, :]  # Add batch dimension
                    
                    rng_key, action_key = jax.random.split(rng_key)
                    network_output = player.apply_fn(player.params, obs_array)
                    
                    # Handle ActorCritic output (pi, value) tuple
                    if isinstance(network_output, tuple):
                        pi, _ = network_output  # Extract policy distribution, ignore value
                        action_logits = pi.logits  # Get logits from distrax.Categorical
                    else:
                        action_logits = network_output  # Direct logits output
                    
                    action = jax.random.categorical(action_key, action_logits).squeeze()
                    action = int(action)
                
                actions[agent] = action
                if step_actions is not None:
                    step_actions[agent] = action
            
            # Log step data
            if observations_log is not None:
                observations_log.append({agent: obs[agent].tolist() for agent in self.env.agents})
            if actions_log is not None:
                actions_log.append(step_actions)
            
            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, actions)
            
            # Update rewards
            for agent in self.env.agents:
                episode_rewards[agent] += float(rewards[agent])
            
            if rewards_log is not None:
                rewards_log.append({agent: float(rewards[agent]) for agent in self.env.agents})
            
            step_count += 1
            done = any(dones.values()) or step_count >= self.max_episode_steps
        
        # Determine winner and termination reason
        agent_0, agent_1 = self.env.agents[0], self.env.agents[1]
        final_rewards = {agent_0: float(rewards[agent_0]), agent_1: float(rewards[agent_1])}
        
        if step_count >= self.max_episode_steps:
            termination_reason = "max_steps"
            if final_rewards[agent_0] > final_rewards[agent_1]:
                winner = player1.name
            elif final_rewards[agent_1] > final_rewards[agent_0]:
                winner = player2.name
            else:
                winner = None  # Draw
        else:
            termination_reason = "winner"
            if final_rewards[agent_0] > final_rewards[agent_1]:
                winner = player1.name
            elif final_rewards[agent_1] > final_rewards[agent_0]:
                winner = player2.name
            else:
                winner = None
                termination_reason = "draw"
        
        # Create episode data
        episode_data = EpisodeData(
            episode_id=episode_id,
            match_id=match_id,
            player1_name=player1.name,
            player2_name=player2.name,
            player1_position="agent_0",
            player2_position="agent_1",
            winner=winner,
            episode_length=step_count,
            player1_total_reward=episode_rewards[agent_0],
            player2_total_reward=episode_rewards[agent_1],
            player1_final_reward=final_rewards[agent_0],
            player2_final_reward=final_rewards[agent_1],
            termination_reason=termination_reason,
            timestamp=datetime.now().isoformat(),
            observations=observations_log,
            actions=actions_log,
            rewards_sequence=rewards_log
        )
        
        return episode_data
    
    def run_match_detailed(self, player1: TournamentPlayer, player2: TournamentPlayer, rng_key):
        """Run a complete match with detailed data collection."""
        
        match_id = f"{player1.name}_vs_{player2.name}"
        print(f"🥊 Running detailed match: {match_id}")
        
        # Load checkpoint players if needed
        for player in [player1, player2]:
            if player.player_type == 'checkpoint' and player.params is None:
                self.evaluator.load_checkpoint_player(player)
                
            # Verify player is properly loaded
            if player.player_type == 'checkpoint' and (player.params is None or player.apply_fn is None):
                raise ValueError(f"Failed to load checkpoint player {player.name}: params={player.params is not None}, apply_fn={player.apply_fn is not None}")
        
        episodes = []
        episode_id = 0
        
        # Run episodes with player1 as agent_0, player2 as agent_1
        for _ in range(self.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            episode_data = self.run_single_episode_detailed(
                player1, player2, episode_id, match_id, episode_key
            )
            episodes.append(episode_data)
            episode_id += 1
        
        # Run episodes with positions swapped
        for _ in range(self.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            episode_data = self.run_single_episode_detailed(
                player2, player1, episode_id, match_id, episode_key
            )
            # Adjust for position swap
            episode_data.player1_name = player1.name
            episode_data.player2_name = player2.name
            episode_data.player1_position = "agent_1"
            episode_data.player2_position = "agent_0"
            # Swap rewards and winner
            episode_data.player1_total_reward, episode_data.player2_total_reward = \
                episode_data.player2_total_reward, episode_data.player1_total_reward
            episode_data.player1_final_reward, episode_data.player2_final_reward = \
                episode_data.player2_final_reward, episode_data.player1_final_reward
            if episode_data.winner == player2.name:
                episode_data.winner = player1.name
            elif episode_data.winner == player1.name:
                episode_data.winner = player2.name
            
            episodes.append(episode_data)
            episode_id += 1
        
        # Calculate match statistics
        player1_wins = sum(1 for ep in episodes if ep.winner == player1.name)
        player2_wins = sum(1 for ep in episodes if ep.winner == player2.name)
        draws = sum(1 for ep in episodes if ep.winner is None)
        total_episodes = len(episodes)
        
        match_data = MatchData(
            match_id=match_id,
            player1_name=player1.name,
            player2_name=player2.name,
            total_episodes=total_episodes,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            player1_win_rate=player1_wins / total_episodes,
            player2_win_rate=player2_wins / total_episodes,
            draw_rate=draws / total_episodes,
            avg_episode_length=np.mean([ep.episode_length for ep in episodes]),
            player1_avg_reward=np.mean([ep.player1_total_reward for ep in episodes]),
            player2_avg_reward=np.mean([ep.player2_total_reward for ep in episodes]),
            episodes=episodes
        )
        
        # Save match data immediately
        self.save_match_data(match_data)
        
        return match_data
    
    def save_match_data(self, match_data: MatchData):
        """Save detailed match data to files."""
        
        # Save match summary as JSON
        summary_file = self.match_summaries_dir / f"{match_data.match_id}_summary.json"
        summary_dict = asdict(match_data)
        # Remove episodes from summary (too large)
        summary_dict['episodes'] = f"{len(match_data.episodes)} episodes (see raw data)"
        
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Save detailed episode data
        episodes_file = self.raw_data_dir / f"{match_data.match_id}_episodes.json"
        episodes_data = [self._convert_jax_arrays_to_lists(asdict(ep)) for ep in match_data.episodes]
        
        with open(episodes_file, 'w') as f:
            json.dump(episodes_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_file = self.raw_data_dir / f"{match_data.match_id}_episodes.csv"
        with open(csv_file, 'w', newline='') as f:
            if match_data.episodes:
                # Get fieldnames excluding complex nested data
                fieldnames = [field for field in asdict(match_data.episodes[0]).keys() 
                             if field not in ['observations', 'actions', 'rewards_sequence']]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for episode in match_data.episodes:
                    row = asdict(episode)
                    # Remove complex nested data from CSV
                    for field in ['observations', 'actions', 'rewards_sequence']:
                        row.pop(field, None)
                    writer.writerow(row)
        
        print(f"💾 Saved match data: {match_data.match_id}")
    
    def _convert_jax_arrays_to_lists(self, data):
        """Recursively convert JAX arrays to Python lists for JSON serialization."""
        import jax.numpy as jnp
        
        if isinstance(data, dict):
            return {key: self._convert_jax_arrays_to_lists(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_jax_arrays_to_lists(item) for item in data]
        elif hasattr(data, '__array__') and hasattr(data, 'tolist'):  # JAX array
            return data.tolist()
        else:
            return data
    
    def run_comprehensive_tournament(self, rng_key):
        """Run comprehensive tournament with detailed data collection."""
        
        print(f"\n🏆 Starting Comprehensive Tournament Data Collection")
        print(f"📊 Total players: {len(self.players)}")
        print(f"🥊 Total matches: {len(list(itertools.combinations(self.players, 2)))}")
        print(f"📈 Episodes per match: {self.episodes_per_matchup}")
        
        start_time = time.time()
        match_count = 0
        total_matches = len(list(itertools.combinations(self.players, 2)))
        
        # Run all matchups (including scripted vs scripted)
        for player1, player2 in itertools.combinations(self.players, 2):
            match_count += 1
            print(f"\n🎯 Match {match_count}/{total_matches}: {player1.name} vs {player2.name}")
            
            rng_key, match_key = jax.random.split(rng_key)
            match_data = self.run_match_detailed(player1, player2, match_key)
            self.match_data.append(match_data)
            
            # Add episodes to global list
            self.episode_data.extend(match_data.episodes)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_match = elapsed / match_count
            remaining_matches = total_matches - match_count
            eta = remaining_matches * avg_time_per_match
            
            print(f"⏱️  Match completed in {time.time() - start_time:.1f}s")
            print(f"📊 Progress: {match_count}/{total_matches} ({match_count/total_matches*100:.1f}%)")
            print(f"🕐 ETA: {eta/60:.1f} minutes")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Tournament completed in {total_time/60:.1f} minutes!")
        print(f"📊 Total episodes collected: {len(self.episode_data)}")
        
        # Save comprehensive summary
        self.save_tournament_summary()
    
    def save_tournament_summary(self):
        """Save comprehensive tournament summary."""
        
        summary_file = self.output_dir / "tournament_summary.json"
        
        # Calculate overall statistics
        total_episodes = len(self.episode_data)
        total_matches = len(self.match_data)
        
        # Player performance summary
        player_stats = {}
        for player in self.players:
            wins = sum(1 for ep in self.episode_data if ep.winner == player.name)
            total_games = sum(1 for ep in self.episode_data 
                            if ep.player1_name == player.name or ep.player2_name == player.name)
            win_rate = wins / total_games if total_games > 0 else 0
            
            player_stats[player.name] = {
                'wins': wins,
                'total_games': total_games,
                'win_rate': win_rate,
                'player_type': player.player_type,
                'algorithm': player.algorithm if hasattr(player, 'algorithm') else None
            }
        
        summary = {
            'tournament_info': {
                'timestamp': datetime.now().isoformat(),
                'environment': self.env_name,
                'episodes_per_matchup': self.episodes_per_matchup,
                'max_episode_steps': self.max_episode_steps,
                'total_players': len(self.players),
                'total_matches': total_matches,
                'total_episodes': total_episodes
            },
            'data_collection_settings': {
                'include_observations': self.include_observations,
                'include_actions': self.include_actions,
                'include_rewards_sequence': self.include_rewards_sequence
            },
            'player_performance': player_stats,
            'files_generated': {
                'raw_episode_data': f"{total_matches} JSON files in raw_data/",
                'match_summaries': f"{total_matches} JSON files in match_summaries/",
                'csv_files': f"{total_matches} CSV files in raw_data/"
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Tournament summary saved: {summary_file}")
        
        # Also save a consolidated CSV with all episodes
        all_episodes_csv = self.output_dir / "all_episodes.csv"
        if self.episode_data:
            fieldnames = [field for field in asdict(self.episode_data[0]).keys() 
                         if field not in ['observations', 'actions', 'rewards_sequence']]
            
            with open(all_episodes_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for episode in self.episode_data:
                    row = asdict(episode)
                    # Remove complex nested data from CSV
                    for field in ['observations', 'actions', 'rewards_sequence']:
                        row.pop(field, None)
                    writer.writerow(row)
        
        print(f"📊 All episodes CSV saved: {all_episodes_csv}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive tournament data collection")
    parser.add_argument("--episodes-per-matchup", type=int, default=100,
                       help="Number of episodes per matchup (default: 100)")
    parser.add_argument("--output-dir", type=str, default="tournament_data",
                       help="Output directory for data (default: tournament_data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--latest-only", action="store_true", default=True,
                       help="Only use latest checkpoints (default: True)")
    parser.add_argument("--include-observations", action="store_true",
                       help="Include observation data (increases file size)")
    parser.add_argument("--include-actions", action="store_true", default=True,
                       help="Include action data (default: True)")
    parser.add_argument("--include-rewards-sequence", action="store_true", default=True,
                       help="Include reward sequence data (default: True)")
    
    args = parser.parse_args()
    
    # Initialize tournament evaluator first
    evaluator = TournamentEvaluator(
        episodes_per_matchup=args.episodes_per_matchup,
        output_dir="temp_tournament_results",  # Temporary directory for evaluator
    )
    
    # Initialize collector with evaluator
    collector = ComprehensiveTournamentDataCollector(
        evaluator=evaluator,
        episodes_per_matchup=args.episodes_per_matchup,
        output_dir=args.output_dir,
        include_observations=args.include_observations,
        include_actions=args.include_actions,
        include_rewards_sequence=args.include_rewards_sequence
    )
    
    # Setup tournament (discover all players)
    latest_only = args.latest_only
    collector.setup_tournament(selected_players=None, latest_only=latest_only)
    
    # Run comprehensive tournament
    rng_key = jax.random.PRNGKey(args.seed)
    collector.run_comprehensive_tournament(rng_key)
    
    print(f"\n🎉 Comprehensive tournament data collection complete!")
    print(f"📁 All data saved in: {args.output_dir}/")
    print(f"\n📊 Next step: Run analysis pipeline on collected data")
    print(f"   python -m baselines.tournament_data_analysis --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
