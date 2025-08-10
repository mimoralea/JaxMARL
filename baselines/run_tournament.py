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
import orbax.checkpoint as ocp
from jaxmarl import make


from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
from baselines.IPPO.train import ActorCritic
from baselines.SPPPO.train import ActorCritic as SPPPOActorCritic
from baselines.FSPPPO.train import ActorCritic as FSPPPOActorCritic


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
                 max_episode_steps: int = 100):
        self.env_name = env_name
        self.episodes_per_matchup = episodes_per_matchup
        self.episodes_per_side = episodes_per_matchup // 2
        self.max_episode_steps = max_episode_steps

        # Create timestamped run folder
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / f"run_{self.run_timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment with fixed starting positions
        if env_name == "MPE_simple_sumo_v3":
            from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
            self.env = SimpleSumoMPE(random_spawn=False)
        else:
            self.env = make(env_name)
        self.env = LogWrapper(self.env)

        # Tournament data
        self.players = []
        self.matches = []
        self.results = []

        print("Tournament Evaluator initialized")
        print(f"Environment: {env_name}")
        print(f"Episodes per matchup: {episodes_per_matchup} "
              f"({self.episodes_per_side} per side)")
        print(f"Max episode steps: {max_episode_steps}")
        print(f"Output directory: {self.output_dir}")

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
        """Get the most recent IPPO checkpoint."""

        ippo_runs = glob.glob("training_runs/IPPO_*/")
        if not ippo_runs:
            return []

        latest_run = max(ippo_runs, key=os.path.getmtime)

        # Extract seed from path
        match = re.search(r'seed_(\d+)', latest_run)
        seed = int(match.group(1)) if match else 0

        return [TournamentPlayer(
            name=f"IPPO_latest_seed{seed}",
            player_type='checkpoint',
            checkpoint_path=latest_run,
            algorithm='IPPO',
            seed=seed
        )]

    def _get_latest_spppo_checkpoint(self):
        """Get the most recent SPPPO checkpoint."""

        spppo_runs = glob.glob("training_runs/SPPPO_*/")
        if not spppo_runs:
            return []

        latest_run = max(spppo_runs, key=os.path.getmtime)

        # Extract seed from path
        match = re.search(r'seed_(\d+)', latest_run)
        seed = int(match.group(1)) if match else 0

        return [TournamentPlayer(
            name=f"SPPPO_latest_seed{seed}",
            player_type='checkpoint',
            checkpoint_path=latest_run,
            algorithm='SPPPO',
            seed=seed
        )]

    def _get_latest_fspppo_checkpoint(self):
        """Get the most recent FSPPPO checkpoint."""
        fspppo_pattern = "checkpoints/fspppo/run_*_seed*/main_agent/step_*/"
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
                checkpoint_path=latest_path,
                seed=seed
            )]

        return []

    def _get_all_ippo_checkpoints(self):
        """Get all IPPO checkpoints (original behavior)."""
        players = []
        # Check both main and opponent directories for IPPO
        for agent_type in ['main', 'opponent']:
            ippo_pattern = f"checkpoints/ippo/run_*_seed*/{agent_type}/"
            ippo_paths = glob.glob(ippo_pattern)

            for path in ippo_paths:
                if os.path.isdir(path):
                    # Extract run info
                    parts = path.split('/')
                    run_seed = parts[-2]  # e.g., "run_20250728_105631_seed0"
                    agent_id = agent_type  # "main" or "opponent"

                    # Extract seed number
                    seed_match = re.search(r'seed(\d+)', run_seed)
                    seed = int(seed_match.group(1)) if seed_match else 0

                    # Find latest checkpoint in this agent directory
                    # IPPO uses Orbax with numeric directories (e.g., '4882.0')
                    checkpoint_dirs = [
                        d for d in glob.glob(os.path.join(path, "*"))
                        if (os.path.isdir(d) and
                            os.path.basename(d).replace('.', '').isdigit())]
                    if checkpoint_dirs:
                        latest_checkpoint = max(
                            checkpoint_dirs,
                            key=lambda x: float(os.path.basename(x))
                        )
                        step = os.path.basename(latest_checkpoint).split('.')[0]
                        name = (f"IPPO_{run_seed.split('_')[-1]}_"
                                f"{agent_id}_step{step}")
                        players.append(TournamentPlayer(
                            name=name,
                            player_type='checkpoint',
                            algorithm='IPPO',
                            checkpoint_path=latest_checkpoint,
                            seed=seed
                        ))

        return players

    def _get_training_setting_pairs(self, latest_only=False):
        """Get training setting pairs for each algorithm type.

        Returns players configured for training setting evaluation:
        - IPPO: agent_0 vs agent_1 from same training run.
        - SPPPO: shared_agent vs shared_agent (self-play).
        - FSPPPO: main_agent vs main_agent (self-play).
        """
        training_players = []

        # IPPO Training Setting: main vs opponent from same run
        ippo_pattern = "checkpoints/ippo/run_*_seed*/*/"
        ippo_paths = glob.glob(ippo_pattern)

        # Group by run_seed to find agent pairs
        run_groups = {}
        for path in ippo_paths:
            if os.path.isdir(path):
                parts = path.split('/')
                run_seed = parts[-2]
                agent_id = parts[-1]

                if run_seed not in run_groups:
                    run_groups[run_seed] = {}

                # Find latest checkpoint in this agent directory
                checkpoint_dirs = [
                    d for d in glob.glob(os.path.join(path, "*"))
                    if (os.path.isdir(d) and
                        os.path.basename(d).replace('.', '').isdigit())
                ]
                if checkpoint_dirs:
                    latest_checkpoint = max(
                        checkpoint_dirs,
                        key=lambda x: float(os.path.basename(x))
                    )
                    run_groups[run_seed][agent_id] = latest_checkpoint

        # Find the most recent run with both agents
        latest_run = None
        latest_time = 0
        for run_seed, agents in run_groups.items():
            if 'agent_0' in agents and 'agent_1' in agents:
                # Use the modification time of agent_0's checkpoint
                mtime = os.path.getmtime(agents['agent_0'])
                if mtime > latest_time:
                    latest_time = mtime
                    latest_run = run_seed

        if latest_run:
            seed_match = re.search(r'seed(\d+)', latest_run)
            seed = int(seed_match.group(1)) if seed_match else 0

            # Add both agents from the training pair
            for agent_id in ['agent_0', 'agent_1']:
                name = f"IPPO_training_{agent_id}"
                training_players.append(TournamentPlayer(
                    name=name,
                    player_type='checkpoint',
                    algorithm='IPPO',
                    checkpoint_path=run_groups[latest_run][agent_id],
                    seed=seed
                ))

        # SPPPO Training Setting: shared_agent vs shared_agent (self-play)
        spppo_latest = self._get_latest_spppo_checkpoint()
        if spppo_latest:
            # Create a duplicate for self-play evaluation
            original = spppo_latest[0]
            training_players.append(TournamentPlayer(
                name="SPPPO_training_main",
                player_type='checkpoint',
                algorithm='SPPPO',
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            ))
            opponent_spppo = TournamentPlayer(
                name="SPPPO_training_opponent",
                player_type='checkpoint',
                algorithm='SPPPO',
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            )
            training_players.append(opponent_spppo)

        # FSPPPO Training Setting: main_agent vs main_agent (self-play)
        fspppo_latest = self._get_latest_fspppo_checkpoint()
        if fspppo_latest:
            # Create a duplicate for self-play evaluation
            original = fspppo_latest[0]
            training_players.append(TournamentPlayer(
                name="FSPPPO_training_main",
                player_type='checkpoint',
                algorithm='FSPPPO',
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            ))
            opponent_fspppo = TournamentPlayer(
                name="FSPPPO_training_opponent",
                player_type='checkpoint',
                algorithm='FSPPPO',
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            )
            training_players.append(opponent_fspppo)

        return training_players

    def _get_all_spppo_checkpoints(self):
        """Get all SPPPO checkpoints (original behavior)."""
        players = []
        spppo_pattern = "checkpoints/spppo/run_*_seed*/shared_agent/*/"
        spppo_paths = glob.glob(spppo_pattern)

        for path in spppo_paths:
            if os.path.isdir(path):
                parts = path.split('/')
                run_seed = parts[-3]
                seed_match = re.search(r'seed(\d+)', run_seed)
                seed = int(seed_match.group(1)) if seed_match else 0

                checkpoint_dirs = [
                    d for d in glob.glob(os.path.join(path, "*"))
                    if (os.path.isdir(d) and
                        os.path.basename(d).replace('.', '').isdigit())]
                if checkpoint_dirs:
                    latest_checkpoint = max(
                        checkpoint_dirs,
                        key=lambda x: float(os.path.basename(x))
                    )
                    step = os.path.basename(latest_checkpoint).split('.')[0]
                    name = f"SPPPO_{run_seed.split('_')[-1]}_step{step}"
                    players.append(TournamentPlayer(
                        name=name,
                        player_type='checkpoint',
                        algorithm='SPPPO',
                        checkpoint_path=latest_checkpoint,
                        seed=seed
                    ))
        return players

    def _get_all_fspppo_checkpoints(self):
        """Get all FSPPPO checkpoints (original behavior)."""
        players = []
        fspppo_pattern = "checkpoints/fspppo/run_*_seed*/main_agent/step_*"
        fspppo_checkpoints = glob.glob(fspppo_pattern)
        for checkpoint_dir in fspppo_checkpoints:
            path_parts = checkpoint_dir.strip('/').split('/')
            run_seed_dir = path_parts[-3]
            seed_match = re.search(r'seed(\d+)', run_seed_dir)
            seed = int(seed_match.group(1)) if seed_match else 0
            step = int(path_parts[-1].split('_')[-1])
            name = f"FSPPPO_seed{seed}_step{step}"
            players.append(TournamentPlayer(
                name=name,
                player_type='checkpoint',
                checkpoint_path=checkpoint_dir,
                algorithm='FSPPPO',
                seed=seed
            ))

        return players

    def create_scripted_players(self) -> List[TournamentPlayer]:
        """Create scripted opponent players using standardized behaviors."""
        scripted_players = []
        behavior_names = ["noop", "turn-left", "turn-right", "gas", "brake"]
        for name in behavior_names:
            scripted_players.append(TournamentPlayer(
                name=f"scripted_{name}",
                player_type='scripted',
                algorithm='scripted'
            ))
        return scripted_players

    def load_checkpoint_player(self, player: TournamentPlayer):
        """Load parameters and apply function for a checkpoint player."""
        if player.params is not None:
            return
        try:
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

            # Use Orbax to load the checkpoint
            try:
                orbax_checkpointer = ocp.PyTreeCheckpointer()
                restored = orbax_checkpointer.restore(player.checkpoint_path)
                if 'model' in restored:
                    player.params = restored['model']['params']
                else:
                    player.params = restored # Older checkpoints
                player.apply_fn = network.apply
            except Exception as e:
                print(f"Error loading checkpoint for {player.name}: {e}")
                print(f"Path: {player.checkpoint_path}")
                player.params = None
                player.apply_fn = None
        except Exception as e:
            print(f"ERROR loading player {player.name}: {e}")
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

        winner, outcome = self.env.get_info(state)

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
                                  match_id: str) -> Dict[str, Any]:
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
                    actions[self.env.agents[0]] = jax.random.categorical(action_key, pi)
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
                    actions[self.env.agents[1]] = jax.random.categorical(action_key, pi)
                else:
                    actions[self.env.agents[1]] = jax.random.categorical(action_key, network_output)

            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, actions)

            episode_length += 1

            if dones["__all__"] or episode_length >= self.max_episode_steps:
                break

        winner, outcome = self.env.get_info(state)

        green_total_reward = state.rewards[self.env.agents[0]]
        red_total_reward = state.rewards[self.env.agents[1]]

        return {
            "match_id": match_id,
            "episode_id": episode_id,
            "winner": winner,
            "outcome": outcome,
            "steps": state.step,
            "returns": state.rewards,
            "green_player": green_player.name,
            "red_player": red_player.name,
            "green_reward": float(green_total_reward),
            "red_reward": float(red_total_reward),
            "side": side,
        }

    def run_match(self, match: TournamentMatch, rng_key) -> List[Dict[str, Any]]:
        """Run a complete match between two players (symmetrical)."""
        print(f"Running match: {match.player1.name} vs {match.player2.name}")
        print(f"Episodes: {match.total_episodes} "
              f"({match.episodes_per_side} per side)")
        
        match_results = []
        episode_id = 0
        
        # Side 1: Player1 as green (agent_0), Player2 as red (agent_1)
        print(f"Side 1: {match.player1.name} (green) vs "
              f"{match.player2.name} (red)")
        for _ in range(self.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            result = self._run_episode_with_positions(
                green_player=match.player1, red_player=match.player2, 
                rng_key=episode_key, episode_id=episode_id, side=1,
                match_id=match.get_match_id()
            )
            match_results.append(result)
            episode_id += 1
            
            if (episode_id - 1) % 10 == 0:
                print(f"Completed {(episode_id - 1) + 1}/{match.episodes_per_side} episodes")
        
        # Side 2: Player2 as green (agent_0), Player1 as red (agent_1)
        print(f"Side 2: {match.player2.name} (green) vs "
              f"{match.player1.name} (red)")
        for _ in range(self.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            result = self._run_episode_with_positions(
                green_player=match.player2, red_player=match.player1,
                rng_key=episode_key,
                episode_id=episode_id,
                side=2,
                match_id=match.get_match_id(),
            )
            match_results.append(result)
            episode_id += 1
            
            if (episode_id - 1) % 10 == 0:
                print(f"Completed {(episode_id - 1) + 1}/{match.episodes_per_side} episodes")
        
        # Calculate match statistics
        player1_wins = sum(
            1 for r in match_results if r['winner'] == match.player1.name
        )
        player2_wins = sum(
            1 for r in match_results if r['winner'] == match.player2.name
        )
        draws = sum(1 for r in match_results if r['winner'] == 'draw')
        
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
                    [p.name for p in checkpoint_players if 'IPPO' in p.name]
                )
            if 'spppo' in selected_set:
                selected_set.update(
                    [p.name for p in checkpoint_players if 'SPPPO' in p.name]
                )
            if 'fspppo' in selected_set:
                selected_set.update(
                    [p.name for p in checkpoint_players if 'FSPPPO' in p.name]
                )
            if 'scripted' in selected_set:
                selected_set.update([p.name for p in scripted_players])
            
            self.players = [p for p in all_players if p.name in selected_set]
        else:
            self.players = all_players
        
        print(f"Tournament Setup:")
        print(f"Total players: {len(self.players)}")
        
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
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        for i, match in enumerate(self.matches):
            print(f"Match {i+1}/{len(self.matches)}")
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
            'episode_id', 'player1', 'player2', 'player1_reward',
            'player2_reward',
            'player1_color', 'player2_color', 'winner', 'outcome',
            'episode_length', 'completed', 'side',
            'green_player', 'red_player', 'green_reward', 'red_reward'
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
            p1, p2 = result['player1'], result['player2']
            
            # Update episode counts
            player_stats[p1]['episodes'] += 1
            player_stats[p2]['episodes'] += 1
            
            # Update rewards
            player_stats[p1]['total_reward'] += result['player1_reward']
            player_stats[p2]['total_reward'] += result['player2_reward']
            
            # Update win/loss/draw counts
            if result['winner'] == p1:
                player_stats[p1]['wins'] += 1
                player_stats[p2]['losses'] += 1
            elif result['winner'] == p2:
                player_stats[p2]['wins'] += 1
                player_stats[p1]['losses'] += 1
            else:
                player_stats[p1]['draws'] += 1
                player_stats[p2]['draws'] += 1
        
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
        "--players", type=str, default=None,
        help="Comma-separated list of players (e.g., 'ippo,spppo,scripted')"
    )
    parser.add_argument(
        "--episodes-per-matchup", type=int, default=100,
        help="Number of episodes per matchup (default: 100)"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=100,
        help="Maximum steps per episode (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="tournament_results",
        help="Output directory for results (default: tournament_results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
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

    # Initialize tournament
    evaluator = TournamentEvaluator(
        episodes_per_matchup=args.episodes_per_matchup,
        output_dir=args.output_dir,
        max_episode_steps=args.max_episode_steps
    )

    # Setup and run tournament
    latest_only = not args.all_checkpoints
    evaluator.setup_tournament(selected_players, latest_only=latest_only)

    # Run tournament
    rng_key = jax.random.PRNGKey(args.seed)
    evaluator.run_tournament(rng_key)

    print("\nTournament evaluation complete!")
    print(f"Results saved in: {evaluator.output_dir}/")


if __name__ == "__main__":
    main()
