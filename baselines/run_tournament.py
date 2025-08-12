#!/usr/bin/env python3
"""
Comprehensive Round-Robin Tournament Evaluation Script

This script runs a complete tournament between all baseline algorithms
(IPPO, SPPPO, FSPPPO) and scripted opponents with proper statistical
analysis and symmetrical matchups.

Features:
- Round-robin tournament format (every player vs every other player)
- Symmetrical matchups (A vs B and B vs A, swapping positions/colors)
- Statistically significant sample sizes (default: 100 episodes per matchup)
- Comprehensive data collection: rewards, outcomes, timesteps-to-win
- CSV export for detailed statistical analysis
- Support for checkpoint discovery and scripted opponent integration
- Configurable player selection (default: all available players)

Usage:
    python -m baselines.tournament_evaluation
    python -m baselines.tournament_evaluation --players ippo,spppo,scripted
    python -m baselines.tournament_evaluation --episodes-per-matchup 200
"""

import argparse
import os
import glob
import csv
import time
import itertools
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxmarl import make


from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
from baselines.IPPO.train import ActorCritic
from baselines.SPPPO.train import ActorCritic as SPPPOActorCritic
from baselines.FSPPPO.train import ActorCritic as FSPPPOActorCritic
from baselines.scripted_behaviors import list_scripted_behaviors


class TournamentPlayer:
    """Represents a player in the tournament."""

    def __init__(self, name: str, player_type: str,
                 checkpoint_path: Optional[str] = None,
                 algorithm: Optional[str] = None, seed: Optional[int] = None):
        self.name = name
        self.player_type = player_type  # 'checkpoint' or 'scripted'
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.seed = seed
        self.params = None
        self.apply_fn = None
        # Diagnostics
        self.param_sum = None  # float fingerprint of parameters
        self.param_count = None  # total number of scalars in params
        self.checkpoint_step = None  # parsed step from path if available

    def __str__(self):
        if self.player_type == 'scripted':
            return f"{self.name} (scripted)"
        return f"{self.name} ({self.algorithm}, seed={self.seed})"


class TournamentMatch:
    """Represents a single match between two players."""

    def __init__(self, player1: TournamentPlayer, player2: TournamentPlayer,
                 episodes_per_side: int = 50):
        self.player1 = player1
        self.player2 = player2
        self.episodes_per_side = episodes_per_side
        self.total_episodes = episodes_per_side * 2  # Symmetrical matchup
        self.results = []

    def get_match_id(self):
        """Generate unique match identifier."""
        return f"{self.player1.name}_vs_{self.player2.name}"


class TournamentEvaluator:
    """Main tournament evaluation system."""

    def __init__(self, env_name: str = "MPE_simple_sumo_v3",
                 episodes_per_matchup: int = 100,
                 output_dir: str = "tournament_results",
                 max_episode_steps: int = 100,
                 training_seeds: list = None,
                 evaluation_seed: int = 0,
                 skip_random_starts: bool = False):
        self.env_name = env_name
        self.episodes_per_matchup = episodes_per_matchup
        self.episodes_per_side = episodes_per_matchup // 2
        self.max_episode_steps = max_episode_steps
        self.training_seeds = training_seeds if training_seeds is not None else [0, 1, 2]
        self.evaluation_seed = evaluation_seed
        self.skip_random_starts = skip_random_starts

        # Create timestamped run folder
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / f"run_{self.run_timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store current training seed for checkpoint discovery (used internally)
        self.current_training_seed = None

        # Progress tracking variables
        self.total_seeds = len(self.training_seeds)
        self.current_seed_index = 0
        self.total_matches_per_seed = 0
        self.current_match_index = 0
        self.total_chunks_in_match = 0
        self.current_chunk_index = 0
        
        # Initialize environment and tournament data structures
        self._initialize_environment_and_data()

    def _print_progress(self):
        """Print three-level progress after each chunk."""
        # Calculate progress percentages
        matchup_progress = (self.current_chunk_index / self.total_chunks_in_match) * 100
        seed_progress = ((self.current_match_index - 1) * self.total_chunks_in_match + self.current_chunk_index) / (self.total_matches_per_seed * self.total_chunks_in_match) * 100
        tournament_progress = ((self.current_seed_index - 1) * self.total_matches_per_seed * self.total_chunks_in_match + 
                              (self.current_match_index - 1) * self.total_chunks_in_match + self.current_chunk_index) / (self.total_seeds * self.total_matches_per_seed * self.total_chunks_in_match) * 100
        
        print(f"Progress - Matchup: {self.current_chunk_index}/{self.total_chunks_in_match} ({matchup_progress:.1f}%)")
        print(f"Progress - Seed {self.current_training_seed}: {((self.current_match_index - 1) * self.total_chunks_in_match + self.current_chunk_index)}/{self.total_matches_per_seed * self.total_chunks_in_match} ({seed_progress:.1f}%)")
        print(f"Progress - Tournament: {((self.current_seed_index - 1) * self.total_matches_per_seed * self.total_chunks_in_match + (self.current_match_index - 1) * self.total_chunks_in_match + self.current_chunk_index)}/{self.total_seeds * self.total_matches_per_seed * self.total_chunks_in_match} ({tournament_progress:.1f}%)")

    def set_current_training_seed(self, seed: int):
        """Set the current training seed for checkpoint discovery."""
        self.current_training_seed = seed
        print(f"Set current training seed to: {seed}")

    def run_multi_seed_tournament(self, selected_players=None, latest_only=False, rng_key=None):
        """Run tournaments for each training seed with separate output files."""
        print(f"\n🎯 Running Multi-Seed Tournament")
        print(f"Training seeds: {self.training_seeds}")
        print(f"Evaluation seed: {self.evaluation_seed}")

        all_results = {}

        for seed_index, seed in enumerate(self.training_seeds):
            self.current_seed_index = seed_index + 1
            print(f"\n{'='*60}")
            print(f"🌱 TRAINING SEED {seed}")
            print(f"{'='*60}")

            # Set current training seed for checkpoint discovery
            self.set_current_training_seed(seed)

            # Create seed-specific output directory
            seed_output_dir = self.output_dir / f"seed_{seed}"
            seed_output_dir.mkdir(parents=True, exist_ok=True)

            # Temporarily update output directory for this seed
            original_output_dir = self.output_dir
            self.output_dir = seed_output_dir

            # Reset tournament state for this seed
            self.players = []
            self.matches = []
            self.results = []

            try:
                # Setup tournament for this specific seed
                self.setup_tournament(selected_players, latest_only=latest_only)

                if not self.players:
                    print(f"⚠️  No players found for training seed {seed}, skipping...")
                    continue

                # Run tournament for this seed
                seed_rng_key, rng_key = jax.random.split(rng_key)
                self.run_tournament(seed_rng_key)

                # Store results for this seed
                all_results[seed] = {
                    'players': len(self.players),
                    'matches': len(self.matches),
                    'episodes': len(self.results),
                    'output_dir': str(seed_output_dir)
                }

                print(f"✅ Training seed {seed} completed: {len(self.results)} episodes")

            except Exception as e:
                print(f"❌ Error with training seed {seed}: {e}")
                all_results[seed] = {'error': str(e)}

            finally:
                # Restore original output directory
                self.output_dir = original_output_dir

        # Generate multi-seed summary
        self._generate_multi_seed_summary(all_results)

        return all_results

    def _generate_multi_seed_summary(self, all_results):
        """Generate a summary of multi-seed tournament results."""
        summary_file = self.output_dir / "multi_seed_tournament_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("Multi-Seed Tournament Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training seeds: {self.training_seeds}\n")
            f.write(f"Evaluation seed: {self.evaluation_seed}\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Episodes per matchup: {self.episodes_per_matchup}\n")
            f.write(f"Skip random starts: {self.skip_random_starts}\n\n")

            total_episodes = 0
            successful_seeds = 0

            for seed in self.training_seeds:
                f.write(f"Training Seed {seed}:\n")
                if seed in all_results:
                    result = all_results[seed]
                    if 'error' in result:
                        f.write(f"  ❌ Error: {result['error']}\n")
                    else:
                        f.write(f"  ✅ Players: {result['players']}\n")
                        f.write(f"  ✅ Matches: {result['matches']}\n")
                        f.write(f"  ✅ Episodes: {result['episodes']}\n")
                        f.write(f"  ✅ Output: {result['output_dir']}\n")
                        total_episodes += result['episodes']
                        successful_seeds += 1
                else:
                    f.write(f"  ⚠️  No results\n")
                f.write("\n")

            f.write(f"Summary:\n")
            f.write(f"  Total successful seeds: {successful_seeds}/{len(self.training_seeds)}\n")
            f.write(f"  Total episodes: {total_episodes}\n")
            f.write(f"  Results saved in separate directories per seed\n")

        print(f"\n📊 Multi-seed summary saved to: {summary_file}")

    def _initialize_environment_and_data(self):
        """Initialize environment and tournament data structures."""
        # Initialize environment; default to deterministic spawns
        if self.env_name == "MPE_simple_sumo_v3":
            from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
            self.env = SimpleSumoMPE(random_spawn=False)
        else:
            self.env = make(self.env_name)
        self.env = LogWrapper(self.env)

        # Tournament data
        self.players = []
        self.matches = []
        self.results = []

        print("Tournament Evaluator initialized")
        print(f"Environment: {self.env_name}")
        print(f"Episodes per matchup: {self.episodes_per_matchup} "
              f"({self.episodes_per_side} per side)")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Output directory: {self.output_dir}")
        print(f"Skip random starts: {self.skip_random_starts}")

    def _set_spawn_mode(self, random_mode: bool):
        """Set environment spawn mode for MPE Simple Sumo.

        Re-instantiates the env to ensure spawn behavior takes effect.
        """
        if self.env_name == "MPE_simple_sumo_v3":
            from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
            # Recreate environment with desired spawn mode and re-wrap
            self.env = LogWrapper(SimpleSumoMPE(random_spawn=random_mode))
        else:
            # Best-effort toggle if supported
            try:
                if hasattr(self.env, 'random_spawn'):
                    setattr(self.env, 'random_spawn', random_mode)
            except Exception:
                pass

    def _run_batch_episodes(self, green_player: TournamentPlayer,
                           red_player: TournamentPlayer,
                           rng_key, num_episodes: int,
                           side: int, match_id: str,
                           spawn_mode: str = "deterministic",
                           batch_size: int = 10) -> List[Dict[str, Any]]:
        """Run episodes in batches for better performance."""

        # Load player parameters if needed
        if green_player.player_type == 'checkpoint' and green_player.params is None:
            self.load_checkpoint_player(green_player)
        if red_player.player_type == 'checkpoint' and red_player.params is None:
            self.load_checkpoint_player(red_player)

        all_results = []
        episode_id_start = 0

        # Process episodes in batches
        for batch_start in range(0, num_episodes, batch_size):
            batch_end = min(batch_start + batch_size, num_episodes)
            current_batch_size = batch_end - batch_start

            # Split RNG keys for this batch
            batch_keys = jax.random.split(rng_key, current_batch_size + 1)
            rng_key = batch_keys[0]  # Update for next batch
            episode_keys = batch_keys[1:]

            # Run episodes in this batch sequentially (still faster due to reduced overhead)
            batch_results = []
            for i in range(current_batch_size):
                result = self._run_episode_with_positions(
                    green_player=green_player,
                    red_player=red_player,
                    rng_key=episode_keys[i],
                    episode_id=episode_id_start + i,
                    side=side,
                    match_id=match_id,
                    spawn_mode=spawn_mode
                )
                batch_results.append(result)

            all_results.extend(batch_results)
            episode_id_start += current_batch_size

            # Print progress for long runs
            if num_episodes > 20:
                print(f"    Completed {batch_end}/{num_episodes} episodes")

        # Calculate summary statistics for this batch
        green_wins = 0
        red_wins = 0
        draws = 0
        total_green_reward = 0.0
        total_red_reward = 0.0
        
        for result in all_results:
            if result["winner"] == self.env.agents[0]:  # green
                green_wins += 1
            elif result["winner"] == self.env.agents[1]:  # red
                red_wins += 1
            else:  # draw
                draws += 1
            
            total_green_reward += result["green_reward"]
            total_red_reward += result["red_reward"]
        
        # Print summary
        print(f"\n    BLOCK SUMMARY ({spawn_mode} spawn mode):")
        print(f"    {green_player.name} (green): {green_wins}/{num_episodes} wins ({green_wins/num_episodes:.2f}), avg reward: {total_green_reward/num_episodes:.3f}")
        print(f"    {red_player.name} (red): {red_wins}/{num_episodes} wins ({red_wins/num_episodes:.2f}), avg reward: {total_red_reward/num_episodes:.3f}")
        print(f"    Draws: {draws}/{num_episodes} ({draws/num_episodes:.2f})\n")
        
        # Update chunk progress and display progress
        self.current_chunk_index += 1
        self._print_progress()
        print()  # Extra line for spacing
        
        return all_results

    def _run_optimized_match_chunk(self, green_player: TournamentPlayer,
                                  red_player: TournamentPlayer,
                                  num_episodes: int, side: int,
                                  match_id: str, spawn_mode: str,
                                  rng_key, start_episode_id: int = 0) -> List[Dict[str, Any]]:
        """Run a chunk of episodes with optimized batching."""

        if num_episodes == 0:
            return []

        # Use batch processing for better performance
        batch_size = min(10, num_episodes)  # Adjust batch size based on available memory

        return self._run_batch_episodes(
            green_player=green_player,
            red_player=red_player,
            rng_key=rng_key,
            num_episodes=num_episodes,
            side=side,
            match_id=match_id,
            spawn_mode=spawn_mode,
            batch_size=batch_size
        )

    def discover_checkpoint_players(self, latest_only=False,
                                  include_training_pairs=False):
        """Discover available checkpoint players from trained models.

        Args:
            latest_only: If True, only return the most recent checkpoint for
                         each algorithm type.
            include_training_pairs: If True, also include training setting
                                  pairs for each algorithm.
        """
        checkpoint_players = []

        if latest_only:
            # Get only the most recent checkpoint from each algorithm
            checkpoint_players.extend(self._get_latest_ippo_checkpoint())
            checkpoint_players.extend(self._get_latest_spppo_checkpoint())
            checkpoint_players.extend(self._get_latest_fspppo_checkpoint())
        else:
            # Get all checkpoints (original behavior)
            checkpoint_players.extend(self._get_all_ippo_checkpoints())
            checkpoint_players.extend(self._get_all_spppo_checkpoints())
            checkpoint_players.extend(self._get_all_fspppo_checkpoints())

        # Add training setting pairs if requested
        if include_training_pairs:
            training_pairs = self._get_training_setting_pairs(latest_only)
            checkpoint_players.extend(training_pairs)

        print(f"Discovered {len(checkpoint_players)} checkpoint players:")
        for player in checkpoint_players:
            print(f"- {player.name} ({player.algorithm}, seed={player.seed})")

        return checkpoint_players

    def _get_latest_ippo_checkpoint(self):
        """Get the latest IPPO checkpoint for the current training seed."""
        ippo_pattern = f"checkpoints/ippo/run_*_seed{self.current_training_seed}/main/*"
        ippo_paths = glob.glob(ippo_pattern)

        if not ippo_paths:
            return []

        # Find the most recent checkpoint
        latest_path = max(ippo_paths, key=os.path.getmtime)
        parts = latest_path.split('/')
        run_seed = parts[-3]
        seed_match = re.search(r'seed(\d+)', run_seed)
        seed = int(seed_match.group(1)) if seed_match else 0
        step = os.path.basename(latest_path)
        name = f"IPPO_seed{seed}_step{step}"

        return [TournamentPlayer(
            name=name,
            player_type='checkpoint',
            algorithm='IPPO',
            checkpoint_path=os.path.abspath(latest_path),
            seed=seed
        )]

    def _get_latest_spppo_checkpoint(self):
        """Get the latest SPPPO checkpoint for the current training seed."""
        spppo_pattern = f"checkpoints/spppo/run_*_seed{self.current_training_seed}/main/*"
        spppo_paths = glob.glob(spppo_pattern)

        if not spppo_paths:
            return []

        # Find the most recent checkpoint
        latest_path = max(spppo_paths, key=os.path.getmtime)
        parts = latest_path.split('/')
        run_seed = parts[-3]
        seed_match = re.search(r'seed(\d+)', run_seed)
        seed = int(seed_match.group(1)) if seed_match else 0
        step = os.path.basename(latest_path)
        name = f"SPPPO_seed{seed}_step{step}"

        return [TournamentPlayer(
            name=name,
            player_type='checkpoint',
            algorithm='SPPPO',
            checkpoint_path=os.path.abspath(latest_path),
            seed=seed
        )]

    def _get_latest_fspppo_checkpoint(self):
        """Get the most recent FSPPPO checkpoint for the current training seed."""
        fspppo_pattern = f"checkpoints/fspppo/run_*_seed{self.current_training_seed}/main/*"
        fspppo_paths = glob.glob(fspppo_pattern)

        if not fspppo_paths:
            return []

        # Find the most recent checkpoint
        latest_path = None
        latest_time = 0

        for path in fspppo_paths:
            if os.path.isdir(path):
                mtime = os.path.getmtime(path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_path = path

        if latest_path:
            parts = latest_path.split('/')
            run_seed = parts[-3]
            seed_match = re.search(r'seed(\d+)', run_seed)
            seed = int(seed_match.group(1)) if seed_match else 0

            name = "FSPPPO_latest"
            return [TournamentPlayer(
                name=name,
                player_type='checkpoint',
                algorithm='FSPPPO',
                checkpoint_path=os.path.abspath(latest_path),
                seed=seed
            )]

        return []

    def _get_all_ippo_checkpoints(self):
        """Get all IPPO checkpoints for the specified training seed."""
        players = []
        ippo_pattern = f"checkpoints/ippo/run_*_seed{self.training_seed}/main/*"
        ippo_paths = glob.glob(ippo_pattern)

        for path in ippo_paths:
            if os.path.isdir(path):
                parts = path.split('/')
                run_seed = parts[-3]
                seed_match = re.search(r'seed(\d+)', run_seed)
                seed = int(seed_match.group(1)) if seed_match else 0
                step = os.path.basename(path)
                name = f"IPPO_seed{seed}_step{step}"
                players.append(TournamentPlayer(
                    name=name,
                    player_type='checkpoint',
                    algorithm='IPPO',
                    checkpoint_path=os.path.abspath(path),
                    seed=seed
                ))
        return players

    def _get_all_spppo_checkpoints(self):
        """Get all SPPPO checkpoints for the specified training seed."""
        players = []
        spppo_pattern = f"checkpoints/spppo/run_*_seed{self.training_seed}/main/*"
        spppo_paths = glob.glob(spppo_pattern)

        for path in spppo_paths:
            if os.path.isdir(path):
                parts = path.split('/')
                run_seed = parts[-3]
                seed_match = re.search(r'seed(\d+)', run_seed)
                seed = int(seed_match.group(1)) if seed_match else 0
                step = os.path.basename(path)
                name = f"SPPPO_seed{seed}_step{step}"
                players.append(TournamentPlayer(
                    name=name,
                    player_type='checkpoint',
                    algorithm='SPPPO',
                    checkpoint_path=os.path.abspath(path),
                    seed=seed
                ))
        return players

    def _get_all_fspppo_checkpoints(self):
        """Get all FSPPPO checkpoints for the specified training seed."""
        players = []
        fspppo_pattern = f"checkpoints/fspppo/run_*_seed{self.training_seed}/main/*"
        fspppo_paths = glob.glob(fspppo_pattern)

        for path in fspppo_paths:
            if os.path.isdir(path):
                parts = path.split('/')
                run_seed = parts[-3]
                seed_match = re.search(r'seed(\d+)', run_seed)
                seed = int(seed_match.group(1)) if seed_match else 0
                step = os.path.basename(path)
                name = f"FSPPPO_seed{seed}_step{step}"
                players.append(TournamentPlayer(
                    name=name,
                    player_type='checkpoint',
                    algorithm='FSPPPO',
                    checkpoint_path=os.path.abspath(path),
                    seed=seed
                ))
        return players

    def create_scripted_players(self):
        """Create TournamentPlayer entries for all scripted behaviors.

        Uses `baselines.scripted_behaviors.list_scripted_behaviors()` and
        generates names as `scripted_<behavior>` which downstream logic expects.
        """
        scripted_players = []
        try:
            behavior_names = list_scripted_behaviors()
        except Exception:
            # Fallback to a safe minimal set if discovery fails
            behavior_names = []

        for name in behavior_names:
            scripted_players.append(TournamentPlayer(
                name=f"scripted_{name}",
                player_type='scripted',
                algorithm=name,
            ))
        return scripted_players

    def get_scripted_action(self, obs, player_name, rng_key):
        """Get action from scripted behavior."""
        from baselines.scripted_behaviors import get_scripted_action
        # Extract behavior name from player name (remove "scripted_" prefix)
        behavior_name = player_name.replace("scripted_", "")
        return get_scripted_action(obs, behavior_name, rng_key)

    def load_checkpoint_player(self, player: TournamentPlayer):
        """Load parameters and apply function for a checkpoint player."""
        if player.params is not None:
            return

        try:
            # Initialize network based on algorithm
            if player.algorithm == 'IPPO':
                network = ActorCritic(
                    self.env.action_space(self.env.agents[0]).n, activation="tanh"
                )
            elif player.algorithm == 'SPPPO':
                network = SPPPOActorCritic(
                    self.env.action_space(self.env.agents[0]).n, activation="tanh"
                )
            elif player.algorithm == 'FSPPPO':
                network = FSPPPOActorCritic(
                    self.env.action_space(self.env.agents[0]).n, activation="tanh"
                )
            else:
                raise ValueError(f"Unknown algorithm: {player.algorithm}")

            # Use Orbax to load the checkpoint - try different approaches
            try:
                # Try PyTreeCheckpointer first (works for FSPPPO)
                orbax_checkpointer = ocp.PyTreeCheckpointer()
                restored = orbax_checkpointer.restore(player.checkpoint_path)

                # Handle different checkpoint structures
                if 'model' in restored:
                    player.params = restored['model']['params']
                elif 'params' in restored:
                    # FSPPPO stores params directly in the 'params' key
                    # but we need to wrap them for Flax network.apply()
                    if player.algorithm == 'FSPPPO':
                        player.params = {'params': restored['params']}
                    else:
                        player.params = restored['params']
                else:
                    player.params = restored

            except Exception as e1:
                try:
                    # Try loading from train_state subdirectory (IPPO/SPPPO format)
                    train_state_path = os.path.join(player.checkpoint_path, 'train_state')
                    if os.path.exists(train_state_path):
                        orbax_checkpointer = ocp.PyTreeCheckpointer()
                        restored = orbax_checkpointer.restore(train_state_path)
                        player.params = restored['params']
                    else:
                        raise Exception("train_state subdirectory not found")
                except Exception as e2:
                    raise Exception(f"Failed both direct PyTree ({e1}) and train_state loading ({e2})")

            player.apply_fn = network.apply

            # Compute simple parameter fingerprint for diagnostics
            try:
                leaves = jax.tree_util.tree_leaves(player.params)
                total_sum = 0.0
                total_count = 0
                for leaf in leaves:
                    arr = jnp.asarray(leaf)
                    total_sum += float(jnp.sum(arr))
                    total_count += int(arr.size)
                player.param_sum = float(total_sum)
                player.param_count = int(total_count)
            except Exception as _:
                player.param_sum = None
                player.param_count = None

            # Parse checkpoint step from the path if possible
            try:
                base = os.path.basename(os.path.normpath(player.checkpoint_path))
                # supports either numeric steps or prefixed like step_000012
                if base.startswith('step_'):
                    player.checkpoint_step = int(base.split('_')[-1])
                else:
                    player.checkpoint_step = int(base)
            except Exception:
                player.checkpoint_step = None

        except Exception as e:
            print(f"ERROR loading player {player.name}: {e}")
            print(f"Path: {player.checkpoint_path}")
            player.params = None
            player.apply_fn = None

    def run_single_episode(self, player1: TournamentPlayer,
                           player2: TournamentPlayer, rng_key,
                           episode_id: int) -> Dict[str, Any]:
        """Run a single episode between two players."""
        # Reset environment
        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = self.env.reset(reset_key)

        done = {"__all__": False}
        episode_length = 0

        while not done["__all__"]:
            actions = {}
            for agent, player in (
                zip(self.env.agents, [player1, player2])
            ):
                if player.player_type == 'scripted':
                    rng_key, action_key = jax.random.split(rng_key)
                    actions[agent] = self.get_scripted_action(
                        obs[agent], player.name, action_key
                    )
                else:  # Checkpoint player
                    if player.params is None:
                        self.load_checkpoint_player(player)

                    # Get action from neural network
                    rng_key, action_key = jax.random.split(rng_key)
                    network_output = player.apply_fn(player.params, obs[agent])

                    if isinstance(network_output, tuple):
                        pi, _ = network_output
                        if pi.shape[-1] == 1:
                            actions[agent] = pi.squeeze(-1)
                        else:
                            actions[agent] = jax.random.categorical(action_key, pi)
                    else:
                        actions[agent] = jax.random.categorical(action_key, network_output)

            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, actions)

            episode_length += 1

            if dones["__all__"] or episode_length >= self.max_episode_steps:
                break

        # Determine winner based on final rewards
        green_total_reward = cumulative_rewards[self.env.agents[0]]
        red_total_reward = float(rewards[self.env.agents[1]])

        if green_total_reward > red_total_reward:
            winner = self.env.agents[0]
            outcome = "win"
        elif red_total_reward > green_total_reward:
            winner = self.env.agents[1]
            outcome = "win"
        else:
            winner = None
            outcome = "draw"

        return {
            "match_id": f"{player1.name}_vs_{player2.name}",
            "episode_id": episode_id,
            "player1": player1.name,
            "player2": player2.name,
            "winner": winner,
            "outcome": outcome,
            "steps": state.step,
            "returns": state.rewards,
        }

    def _run_episode_with_positions(self, green_player: TournamentPlayer,
                                  red_player: TournamentPlayer,
                                  rng_key, episode_id: int, side: int,
                                  match_id: str, spawn_mode: str = "deterministic") -> Dict[str, Any]:
        """Run a single episode with explicit player position assignments.

        Args:
            green_player: Player assigned to green position (agent_0)
            red_player: Player assigned to red position (agent_1)
            rng_key: Random key for episode
            episode_id: Episode identifier
            side: Indicates which side of the symmetrical matchup this is (1 or 2)
            match_id: The consistent match identifier

        Returns:
            Episode result dictionary with correct winner assignment
        """
        # Reset environment
        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = self.env.reset(reset_key)

        done = {"__all__": False}
        episode_length = 0

        # Initialize cumulative rewards
        cumulative_rewards = {agent: 0.0 for agent in self.env.agents}

        while not done["__all__"]:
            actions = {}

            # Green player (agent_0) action
            if green_player.player_type == 'scripted':
                rng_key, action_key = jax.random.split(rng_key)
                actions[self.env.agents[0]] = self.get_scripted_action(
                    obs[self.env.agents[0]], green_player.name, action_key
                )
            else:  # Checkpoint player
                if green_player.params is None:
                    self.load_checkpoint_player(green_player)

                # Get action from neural network
                rng_key, action_key = jax.random.split(rng_key)
                network_output = green_player.apply_fn(
                    green_player.params, obs[self.env.agents[0]]
                )
                if isinstance(network_output, tuple):
                    pi, _ = network_output
                    # Handle different output types
                    if hasattr(pi, 'sample'):
                        actions[self.env.agents[0]] = pi.sample(seed=action_key)
                    elif hasattr(pi, 'logits'):
                        actions[self.env.agents[0]] = jax.random.categorical(action_key, pi.logits)
                    else:
                        actions[self.env.agents[0]] = jax.random.categorical(action_key, pi)
                else:
                    # Handle different output types
                    if hasattr(network_output, 'sample'):
                        actions[self.env.agents[0]] = network_output.sample(seed=action_key)
                    elif hasattr(network_output, 'logits'):
                        actions[self.env.agents[0]] = jax.random.categorical(action_key, network_output.logits)
                    else:
                        actions[self.env.agents[0]] = jax.random.categorical(action_key, network_output)

            # Red player (agent_1) action
            if red_player.player_type == 'scripted':
                rng_key, action_key = jax.random.split(rng_key)
                actions[self.env.agents[1]] = self.get_scripted_action(
                    obs[self.env.agents[1]], red_player.name, action_key
                )
            else:  # Checkpoint player
                if red_player.params is None:
                    self.load_checkpoint_player(red_player)

                # Get action from neural network
                rng_key, action_key = jax.random.split(rng_key)
                network_output = red_player.apply_fn(
                    red_player.params, obs[self.env.agents[1]]
                )
                if isinstance(network_output, tuple):
                    pi, _ = network_output
                    # Handle different output types
                    if hasattr(pi, 'sample'):
                        actions[self.env.agents[1]] = pi.sample(seed=action_key)
                    elif hasattr(pi, 'logits'):
                        actions[self.env.agents[1]] = jax.random.categorical(action_key, pi.logits)
                    else:
                        actions[self.env.agents[1]] = jax.random.categorical(action_key, pi)
                else:
                    # Handle different output types
                    if hasattr(network_output, 'sample'):
                        actions[self.env.agents[1]] = network_output.sample(seed=action_key)
                    elif hasattr(network_output, 'logits'):
                        actions[self.env.agents[1]] = jax.random.categorical(action_key, network_output.logits)
                    else:
                        actions[self.env.agents[1]] = jax.random.categorical(action_key, network_output)

            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, actions)

            # Accumulate rewards after getting them from the environment
            for agent in self.env.agents:
                cumulative_rewards[agent] += float(rewards[agent])

            episode_length += 1

            if dones["__all__"] or episode_length >= self.max_episode_steps:
                break

        # Determine winner based on cumulative rewards for both players
        green_total_reward = cumulative_rewards[self.env.agents[0]]
        red_total_reward = cumulative_rewards[self.env.agents[1]]

        # No per-episode debug output

        if green_total_reward > red_total_reward:
            winner = self.env.agents[0]  # green
            outcome = "win"
        elif red_total_reward > green_total_reward:
            winner = self.env.agents[1]  # red
            outcome = "win"
        else:
            winner = "draw"
            outcome = "draw"

        return {
            "match_id": match_id,
            "episode_id": episode_id,
            "winner": winner,
            "outcome": outcome,
            "steps": episode_length,
            "returns": cumulative_rewards,  # Use cumulative rewards instead of final step rewards
            "green_player": green_player.name,
            "red_player": red_player.name,
            "green_reward": float(green_total_reward),
            "red_reward": float(red_total_reward),
            "side": side,
            "spawn_mode": spawn_mode,
        }

    def run_match(self, match: TournamentMatch, rng_key) -> List[Dict[str, Any]]:
        """Run a complete match between two players (symmetrical) with optimized batch processing."""
        print(f"Running match: {match.player1.name} vs {match.player2.name}")
        print(f"Episodes: {match.total_episodes} "
              f"({match.episodes_per_side} per side)")

        # Initialize progress tracking for this match
        if self.skip_random_starts:
            self.total_chunks_in_match = 2  # 2 sides only
        else:
            self.total_chunks_in_match = 4  # 2 sides x 2 spawn modes
        self.current_chunk_index = 0

        match_results = []
        episode_id = 0

        if self.skip_random_starts:
            # Deterministic spawns only: two chunks (per side)
            self._set_spawn_mode(random_mode=False)

            # Side 1 deterministic - batch processing
            print(f"Side 1 (deterministic, {self.episodes_per_side} eps): {match.player1.name} (green) vs {match.player2.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side1_results = self._run_optimized_match_chunk(
                green_player=match.player1, red_player=match.player2,
                num_episodes=self.episodes_per_side, side=1,
                match_id=match.get_match_id(), spawn_mode="deterministic",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side1_results)
            episode_id += self.episodes_per_side

            # Side 2 deterministic - batch processing
            print(f"Side 2 (deterministic, {self.episodes_per_side} eps): {match.player2.name} (green) vs {match.player1.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side2_results = self._run_optimized_match_chunk(
                green_player=match.player2, red_player=match.player1,
                num_episodes=self.episodes_per_side, side=2,
                match_id=match.get_match_id(), spawn_mode="deterministic",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side2_results)
            episode_id += self.episodes_per_side
        else:
            # Mixed spawns: four chunks (per side, per spawn mode)
            det_per_side = self.episodes_per_side // 2
            rand_per_side = self.episodes_per_side - det_per_side

            # Deterministic chunks - batch processing
            self._set_spawn_mode(random_mode=False)

            print(f"Side 1 (deterministic, {det_per_side} eps): {match.player1.name} (green) vs {match.player2.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side1_det_results = self._run_optimized_match_chunk(
                green_player=match.player1, red_player=match.player2,
                num_episodes=det_per_side, side=1,
                match_id=match.get_match_id(), spawn_mode="deterministic",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side1_det_results)
            episode_id += det_per_side

            print(f"Side 2 (deterministic, {det_per_side} eps): {match.player2.name} (green) vs {match.player1.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side2_det_results = self._run_optimized_match_chunk(
                green_player=match.player2, red_player=match.player1,
                num_episodes=det_per_side, side=2,
                match_id=match.get_match_id(), spawn_mode="deterministic",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side2_det_results)
            episode_id += det_per_side

            # Random-spawn chunks - batch processing
            self._set_spawn_mode(random_mode=True)

            print(f"Side 1 (random starts, {rand_per_side} eps): {match.player1.name} (green) vs {match.player2.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side1_rand_results = self._run_optimized_match_chunk(
                green_player=match.player1, red_player=match.player2,
                num_episodes=rand_per_side, side=1,
                match_id=match.get_match_id(), spawn_mode="random",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side1_rand_results)
            episode_id += rand_per_side

            print(f"Side 2 (random starts, {rand_per_side} eps): {match.player2.name} (green) vs {match.player1.name} (red)")
            rng_key, chunk_key = jax.random.split(rng_key)
            side2_rand_results = self._run_optimized_match_chunk(
                green_player=match.player2, red_player=match.player1,
                num_episodes=rand_per_side, side=2,
                match_id=match.get_match_id(), spawn_mode="random",
                rng_key=chunk_key, start_episode_id=episode_id
            )
            match_results.extend(side2_rand_results)
            episode_id += rand_per_side

        # Calculate match statistics
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        for r in match_results:
            if r['winner'] == 'draw':
                draws += 1
            elif r['side'] == 1:  # Side 1: player1=green, player2=red
                if r['winner'] == self.env.agents[0]:  # green wins
                    player1_wins += 1
                elif r['winner'] == self.env.agents[1]:  # red wins
                    player2_wins += 1
            else:  # Side 2: player2=green, player1=red
                if r['winner'] == self.env.agents[0]:  # green wins
                    player2_wins += 1
                elif r['winner'] == self.env.agents[1]:  # red wins
                    player1_wins += 1

        print(f"Results: {match.player1.name}: {player1_wins}, "
              f"{match.player2.name}: {player2_wins}, Draws: {draws}")

        return match_results

    def setup_tournament(self, selected_players: Optional[List[str]] = None,
                         latest_only: bool = False):
        """Set up the tournament with all players and matches."""

        # Discover all available players
        checkpoint_players = self.discover_checkpoint_players(latest_only=latest_only)
        scripted_players = self.create_scripted_players()
        all_players = checkpoint_players + scripted_players

        # Filter players if specific ones were requested
        if selected_players:
            selected_set = set(selected_players)
            if 'ippo' in selected_set:
                selected_set.update(
                    [p.name for p in checkpoint_players if p.algorithm == 'IPPO']
                )
            if 'spppo' in selected_set:
                selected_set.update(
                    [p.name for p in checkpoint_players if p.algorithm == 'SPPPO']
                )
            if 'fspppo' in selected_set:
                selected_set.update(
                    [p.name for p in checkpoint_players if p.algorithm == 'FSPPPO']
                )
            if 'scripted' in selected_set:
                selected_set.update([p.name for p in scripted_players])

            # Finalize selected players
            self.players = [
                p for p in all_players if p.name in selected_set
            ]
        else:
            self.players = all_players

        print(f"Tournament Setup:")
        print(f"Total players: {len(self.players)}")

        # Eagerly load checkpoint players once to record diagnostics and manifest
        for p in self.players:
            if p.player_type == 'checkpoint':
                self.load_checkpoint_player(p)

        # Save a manifest for reproducibility/debugging
        try:
            manifest = []
            for p in self.players:
                entry = {
                    'name': p.name,
                    'type': p.player_type,
                    'algorithm': p.algorithm,
                    'seed': p.seed,
                    'checkpoint_path': p.checkpoint_path,
                    'checkpoint_step': p.checkpoint_step,
                    'param_sum': p.param_sum,
                    'param_count': p.param_count,
                }
                manifest.append(entry)
            manifest_path = self.output_dir / 'players_manifest.json'
            with open(manifest_path, 'w') as f:
                import json
                json.dump(manifest, f, indent=2)
            print(f"Saved players manifest: {manifest_path}")
        except Exception as e:
            print(f"Warning: failed to save players manifest: {e}")

        # Create all possible matches (round-robin)
        self.matches = []
        for player1, player2 in itertools.combinations(self.players, 2):
            match = TournamentMatch(player1, player2, self.episodes_per_side)
            self.matches.append(match)

        total_episodes = len(self.matches) * self.episodes_per_matchup
        print(f"\nRunning tournament... (Total matches: {len(self.matches)})")
        print(f"Total episodes: {total_episodes}")

    def run_tournament(self, rng_key):
        """Run the complete tournament."""
        print("Starting Tournament!")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize progress tracking for this tournament
        self.total_matches_per_seed = len(self.matches)
        self.current_match_index = 0

        start_time = time.time()

        for i, match in enumerate(self.matches):
            print(f"Match {i+1}/{len(self.matches)}")
            self.current_match_index = i + 1
            rng_key, match_key = jax.random.split(rng_key)
            match_results = self.run_match(match, match_key)
            self.results.extend(match_results)

            # Save intermediate results every 10 matches
            if (i + 1) % 10 == 0:
                self.save_results(intermediate=True)

        end_time = time.time()
        duration = end_time - start_time

        print("Tournament Complete!")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Total episodes: {len(self.results)}")

        # Save final results
        self.save_results(intermediate=False)
        self.generate_summary()

    def save_results(self, intermediate: bool = False):
        """Save tournament results to CSV."""
        suffix = "_intermediate" if intermediate else ""
        filename = f"tournament_results{suffix}.csv"
        filepath = self.output_dir / filename

        if not self.results:
            print("No results to save")
            return

        fieldnames = [
            'match_id', 'episode_id', 'winner', 'outcome', 'steps', 'returns',
            'green_player', 'red_player', 'green_reward', 'red_reward', 'side', 'spawn_mode'
        ]

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results saved to: {filepath}")

    def generate_summary(self):
        """Generate a summary of tournament results."""
        summary_file = self.output_dir / "tournament_summary.txt"

        # Calculate win rates for each player
        player_stats = {}
        for player in self.players:
            player_stats[player.name] = {
                'wins': 0, 'losses': 0, 'draws': 0, 'total_reward': 0.0,
                'episodes': 0
            }

        for result in self.results:
            green_player = result['green_player']
            red_player = result['red_player']

            # Update episode counts
            player_stats[green_player]['episodes'] += 1
            player_stats[red_player]['episodes'] += 1

            # Update rewards
            player_stats[green_player]['total_reward'] += result['green_reward']
            player_stats[red_player]['total_reward'] += result['red_reward']

            # Update win/loss/draw counts
            if result['winner'] == 'green':
                player_stats[green_player]['wins'] += 1
                player_stats[red_player]['losses'] += 1
            elif result['winner'] == 'red':
                player_stats[red_player]['wins'] += 1
                player_stats[green_player]['losses'] += 1
            else:
                player_stats[green_player]['draws'] += 1
                player_stats[red_player]['draws'] += 1

        # Generate summary
        with open(summary_file, 'w') as f:
            f.write("TOURNAMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Tournament Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Episodes per matchup: {self.episodes_per_matchup}\n")
            f.write(f"Total players: {len(self.players)}\n")
            f.write(f"Total matches: {len(self.matches)}\n")
            f.write(f"Total episodes: {len(self.results)}\n\n")

            f.write("PLAYER STATISTICS\n")
            f.write("-" * 30 + "\n")

            # Sort players by win rate
            sorted_players = sorted(
                player_stats.items(),
                key=lambda x: x[1]['wins'] / max(x[1]['episodes'], 1),
                reverse=True
            )

            for player_name, stats in sorted_players:
                win_rate = stats['wins'] / max(stats['episodes'], 1) * 100
                avg_reward = stats['total_reward'] / max(stats['episodes'], 1)

                f.write(f"\n{player_name}:\n")
                f.write(
                    f"  Win Rate: {win_rate:.1f}% ({stats['wins']}/"
                    f"{stats['episodes']})\n"
                )
                f.write(
                    f"  W/L/D: {stats['wins']}/{stats['losses']}/"
                    f"{stats['draws']}\n"
                )
                f.write(f"  Avg Reward: {avg_reward:.3f}\n")

        print(f"Summary saved to: {summary_file}")


def main():
    """Main tournament execution function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tournament evaluation"
    )
    parser.add_argument(
        "--env", type=str, default="MPE_simple_sumo_v3",
        help="Environment name (default: MPE_simple_sumo_v3)"
    )
    parser.add_argument(
        "--players", type=str, default=None,
        help="Comma-separated list of players (e.g., 'ippo,spppo,scripted')"
    )
    parser.add_argument(
        "--episodes-per-matchup", type=int, default=100,
        help="Number of episodes per matchup (default: 100)"
    )
    parser.add_argument("--max-episode-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument(
        "--output-dir", type=str, default="tournament_results",
        help="Output directory for results (default: tournament_results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    def parse_training_seeds(value):
        """Parse training seeds from CLI argument."""
        # Handle both string and integer inputs
        if isinstance(value, int):
            return [value]

        value_str = str(value)
        if ',' in value_str:
            # Multiple seeds: "0,1,2" or 0,1,2
            return [int(s.strip()) for s in value_str.split(',')]
        else:
            # Single seed: "0" or 0
            return [int(value_str)]

    parser.add_argument(
        "--training-seeds", type=str, default="0,1,2",
        help="Training seeds for checkpoint selection (default: 0,1,2). Can be single seed (e.g., 0 or '0') or multiple seeds (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--evaluation-seed", type=int, default=0,
        help="Evaluation seed for tournament RNG (default: 0)"
    )
    parser.add_argument(
        "--skip-random-starts", action="store_true", default=False,
        help="If set, do not use random-start episodes. When False (default), split each side's episodes into deterministic and random-start chunks."
    )
    parser.add_argument(
        "--all-checkpoints", action="store_true", default=False,
        help="Use all available checkpoints instead of just the latest ones"
    )

    args = parser.parse_args()

    # Parse selected players
    selected_players = None
    if args.players:
        selected_players = [p.strip() for p in args.players.split(',')]

    # Parse training seeds
    training_seeds = parse_training_seeds(args.training_seeds)

    # Initialize tournament
    evaluator = TournamentEvaluator(
        env_name=args.env,
        episodes_per_matchup=args.episodes_per_matchup,
        output_dir=args.output_dir,
        max_episode_steps=args.max_episode_steps,
        training_seeds=training_seeds,
        evaluation_seed=args.evaluation_seed,
        skip_random_starts=args.skip_random_starts
    )

    # Setup and run multi-seed tournament
    latest_only = not args.all_checkpoints
    rng_key = jax.random.PRNGKey(args.evaluation_seed)

    if len(training_seeds) == 1:
        # Single seed mode - use original logic for backward compatibility
        print(f"\n🎯 Running Single-Seed Tournament (seed {training_seeds[0]})")
        evaluator.current_seed_index = 1  # First and only seed
        evaluator.set_current_training_seed(training_seeds[0])
        evaluator.setup_tournament(selected_players, latest_only=latest_only)
        evaluator.run_tournament(rng_key)
        print("\nTournament evaluation complete!")
        print(f"Results saved in: {evaluator.output_dir}/")
    else:
        # Multi-seed mode - run tournaments for each seed separately
        results = evaluator.run_multi_seed_tournament(
            selected_players=selected_players,
            latest_only=latest_only,
            rng_key=rng_key
        )
        print("\nMulti-seed tournament evaluation complete!")
        print(f"Results saved in separate directories under: {evaluator.output_dir}/")

        # Print summary of results
        successful_seeds = sum(1 for r in results.values() if 'error' not in r)
        total_episodes = sum(r.get('episodes', 0) for r in results.values() if 'error' not in r)
        print(f"Successfully completed: {successful_seeds}/{len(training_seeds)} seeds")
        print(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    main()
