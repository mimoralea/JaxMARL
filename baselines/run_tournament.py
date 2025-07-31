#!/usr/bin/env python3
"""
Comprehensive Round-Robin Tournament Evaluation Script

This script runs a complete tournament between all baseline algorithms (IPPO, SPPPO, FSPPPO)
and scripted opponents with proper statistical analysis and symmetrical matchups.

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
import sys
import glob
import csv
import time
import itertools
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from baselines.scripted_behaviors import get_scripted_action, list_scripted_behaviors

# Environment and visualization imports
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper

# Checkpoint loading utilities
try:
    from baselines.IPPO.orbax_checkpoint_manager import IPPOCheckpointManager
except ImportError:
    from IPPO.orbax_checkpoint_manager import IPPOCheckpointManager

try:
    from baselines.SPPPO.orbax_checkpoint_manager import SPPPOCheckpointManager
except ImportError:
    from SPPPO.orbax_checkpoint_manager import SPPPOCheckpointManager

try:
    from baselines.FSPPPO.jax_checkpoint_utils import FSPPPOCheckpointManager
except ImportError:
    from FSPPPO.jax_checkpoint_utils import FSPPPOCheckpointManager


class TournamentPlayer:
    """Represents a player in the tournament."""
    
    def __init__(self, name: str, player_type: str, checkpoint_path: Optional[str] = None, 
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
        else:
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
        self.output_dir = Path(output_dir)
        self.max_episode_steps = max_episode_steps
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
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
        
        print(f"🏆 Tournament Evaluator initialized")
        print(f"   Environment: {env_name}")
        print(f"   Episodes per matchup: {episodes_per_matchup} ({self.episodes_per_side} per side)")
        print(f"   Max episode steps: {max_episode_steps}")
        print(f"   Output directory: {output_dir}")
    
    def discover_checkpoint_players(self, latest_only=False, include_training_pairs=False):
        """Discover available checkpoint players from trained models.
        
        Args:
            latest_only: If True, only return the most recent checkpoint from each algorithm type
            include_training_pairs: If True, also include training setting pairs for each algorithm
        """
        players = []
        
        if latest_only:
            # Get only the most recent checkpoint from each algorithm
            players.extend(self._get_latest_ippo_checkpoint())
            players.extend(self._get_latest_spppo_checkpoint())
            players.extend(self._get_latest_fspppo_checkpoint())
        else:
            # Get all checkpoints (original behavior)
            players.extend(self._get_all_ippo_checkpoints())
            players.extend(self._get_all_spppo_checkpoints())
            players.extend(self._get_all_fspppo_checkpoints())
        
        # Add training setting pairs if requested
        if include_training_pairs:
            training_pairs = self._get_training_setting_pairs(latest_only)
            players.extend(training_pairs)
        
        print(f"📁 Discovered {len(players)} checkpoint players:")
        for player in players:
            print(f"   - {player.name} ({player.algorithm}, seed={player.seed})")
        
        return players
    
    def _get_latest_ippo_checkpoint(self):
        """Get the most recent IPPO checkpoint."""
        ippo_pattern = "checkpoints/ippo/run_*_seed*/main/"
        ippo_paths = glob.glob(ippo_pattern)
        
        if not ippo_paths:
            return []
        
        # Find the most recent checkpoint across all IPPO runs
        latest_path = None
        latest_time = 0
        
        for path in ippo_paths:
            if os.path.isdir(path):
                # IPPO uses Orbax with numeric directories (e.g., '4882.0')
                checkpoint_dirs = [d for d in glob.glob(os.path.join(path, "*")) 
                                 if os.path.isdir(d) and os.path.basename(d).replace('.', '').isdigit()]
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=lambda x: float(os.path.basename(x)))
                    mtime = os.path.getmtime(latest_checkpoint)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_path = path
                        latest_checkpoint_path = latest_checkpoint
        
        if latest_path:
            parts = latest_path.split('/')
            run_seed = parts[-2]
            agent_id = parts[-1]
            seed_match = re.search(r'seed(\d+)', run_seed)
            seed = int(seed_match.group(1)) if seed_match else 0
            
            name = f"IPPO_latest_{agent_id}"
            return [TournamentPlayer(
                name=name,
                player_type='checkpoint',
                algorithm='IPPO',
                checkpoint_path=latest_checkpoint_path,
                seed=seed
            )]
        
        return []
    
    def _get_latest_spppo_checkpoint(self):
        """Get the most recent SPPPO checkpoint."""
        spppo_pattern = "checkpoints/spppo/run_*_seed*/main/*/"
        spppo_paths = glob.glob(spppo_pattern)
        
        if not spppo_paths:
            return []
        
        # Find the most recent checkpoint (filter out non-step directories)
        latest_path = None
        latest_time = 0
        
        for path in spppo_paths:
            if os.path.isdir(path):
                # Filter out duplicate 'shared_agent' directories - only keep step directories
                dirname = os.path.basename(path.rstrip('/'))
                if dirname.replace('.', '').isdigit():  # Only consider numbered step directories
                    mtime = os.path.getmtime(path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_path = path
        
        if latest_path:
            parts = latest_path.split('/')
            run_seed = parts[-3]  # e.g., "run_20250728_111123_seed4"
            seed_match = re.search(r'seed(\d+)', run_seed)
            seed = int(seed_match.group(1)) if seed_match else 0
            
            # SPPPO checkpoint manager expects the base run directory, not the step directory
            # Remove shared_agent/step_dir/'' (last 3 parts) to get base run directory
            base_checkpoint_dir = '/'.join(parts[:-3])
            
            name = "SPPPO_latest"
            return [TournamentPlayer(
                name=name,
                player_type='checkpoint',
                algorithm='SPPPO',
                checkpoint_path=base_checkpoint_dir,
                seed=seed
            )]
        
        return []
    
    def _get_latest_fspppo_checkpoint(self):
        """Get the most recent FSPPPO checkpoint."""
        fspppo_pattern = "checkpoints/fspppo/run_*_seed*/main/*/"
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
                checkpoint_dirs = [d for d in glob.glob(os.path.join(path, "*")) 
                                 if os.path.isdir(d) and os.path.basename(d).replace('.', '').isdigit()]
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=lambda x: float(os.path.basename(x)))
                    
                    name = f"IPPO_{run_seed.split('_')[-1]}_{agent_id}"
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
        - IPPO: agent_0 vs agent_1 from same training run
        - SPPPO: shared_agent vs shared_agent (self-play)
        - FSPPPO: main_agent vs main_agent (self-play)
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
                checkpoint_dirs = [d for d in glob.glob(os.path.join(path, "*")) 
                                 if os.path.isdir(d) and os.path.basename(d).replace('.', '').isdigit()]
                if checkpoint_dirs:
                    latest_checkpoint = max(checkpoint_dirs, key=lambda x: float(os.path.basename(x)))
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
            training_players.append(TournamentPlayer(
                name="SPPPO_training_opponent",
                player_type='checkpoint', 
                algorithm='SPPPO',
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            ))
        
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
            training_players.append(TournamentPlayer(
                name="FSPPPO_training_opponent",
                player_type='checkpoint',
                algorithm='FSPPPO', 
                checkpoint_path=original.checkpoint_path,
                seed=original.seed
            ))
        
        return training_players
    
    def _get_all_spppo_checkpoints(self):
        """Get all SPPPO checkpoints (original behavior)."""
        players = []
        spppo_pattern = "checkpoints/spppo/run_*_seed*/main/*/"
        spppo_checkpoints = glob.glob(spppo_pattern)
        for checkpoint_path in spppo_checkpoints:
            # Extract seed info from path
            path_parts = checkpoint_path.split('/')
            seed_part = [p for p in path_parts if 'seed' in p][0]
            seed = int(seed_part.split('seed')[1])
            
            name = f"SPPPO_seed{seed}"
            players.append(TournamentPlayer(
                name=name,
                player_type='checkpoint',
                checkpoint_path=checkpoint_path,
                algorithm='SPPPO',
                seed=seed
            ))
        
        return players
    
    def _get_all_fspppo_checkpoints(self):
        """Get all FSPPPO checkpoints (original behavior)."""
        players = []
        fspppo_pattern = "checkpoints/fspppo/run_*_seed*/main/"
        fspppo_dirs = glob.glob(fspppo_pattern)
        for checkpoint_dir in fspppo_dirs:
            # Find latest checkpoint in this seed directory
            step_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
            if step_dirs:
                latest_checkpoint = max(step_dirs, key=lambda x: int(x.split('step_')[1]))
                
                # Extract seed info from path
                path_parts = checkpoint_dir.split('/')
                seed_part = [p for p in path_parts if 'seed' in p][0]
                seed = int(seed_part.split('seed')[1])
                
                name = f"FSPPPO_seed{seed}"
                players.append(TournamentPlayer(
                    name=name,
                    player_type='checkpoint',
                    checkpoint_path=latest_checkpoint,
                    algorithm='FSPPPO',
                    seed=seed
                ))
        
        return players
    
    def create_scripted_players(self) -> List[TournamentPlayer]:
        """Create scripted opponent players using standardized behaviors."""
        # Get all available scripted behaviors from the standardized module
        available_behaviors = list_scripted_behaviors()
        players = []
        
        for behavior_name in available_behaviors.keys():
            name = f"scripted_{behavior_name}"
            players.append(TournamentPlayer(
                name=name,
                player_type='scripted',
                algorithm='scripted'
            ))
        
        print(f"🤖 Created {len(players)} scripted players:")
        for player in players:
            print(f"   - {player}")
        
        return players
    
    def load_checkpoint_player(self, player: TournamentPlayer):
        """Load parameters and apply function for a checkpoint player."""
        import os  # Import os at function level to avoid scoping issues
        
        if player.params is not None:
            return  # Already loaded
        
        print(f"📥 Loading checkpoint for {player.name}...")
        
        try:
            if player.algorithm == 'IPPO':
                # Load IPPO checkpoint using Orbax directly
                # checkpoint_path is the full path to the checkpoint directory (e.g., agent_0/4882.0)
                checkpoint_dir = player.checkpoint_path
                
                # Load checkpoint manually using Orbax
                import orbax.checkpoint as ocp
                try:
                    # Directly restore from the checkpoint directory (use absolute path)
                    checkpointer = ocp.PyTreeCheckpointer()
                    abs_checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, 'train_state'))
                    train_state = checkpointer.restore(abs_checkpoint_path)
                    
                    # For IPPO, we need to recreate the network to get apply_fn
                    import sys
                    import jax.numpy as jnp
                    from baselines.IPPO.train import ActorCritic
                    import jaxmarl
                    
                    # Recreate the network (same as in IPPO training)
                    env = jaxmarl.make("MPE_simple_sumo_v3")
                    network = ActorCritic(env.action_space(env.agents[0]).n, activation="tanh")
                    
                    # Handle different checkpoint structures
                    if hasattr(train_state, 'params'):
                        # TrainState object
                        player.params = train_state.params
                        player.apply_fn = network.apply
                    elif isinstance(train_state, dict) and 'params' in train_state:
                        # Dictionary with params
                        player.params = train_state['params']
                        player.apply_fn = network.apply
                    else:
                        # Assume train_state is the params directly
                        player.params = train_state
                        player.apply_fn = network.apply
                    
                    print(f"✅ Loaded IPPO checkpoint from {checkpoint_dir}")
                except Exception as e:
                    print(f"❌ Failed to load IPPO checkpoint: {e}")
                    raise ValueError(f"Failed to load IPPO checkpoint from {checkpoint_dir}: {e}")
                
            elif player.algorithm == 'SPPPO':
                # Load SPPPO checkpoint
                # checkpoint_path is already the base directory for SPPPO
                checkpoint_dir = player.checkpoint_path
                manager = SPPPOCheckpointManager(checkpoint_dir)
                result = manager.load_latest_checkpoint()
                if result is None:
                    # Try to find and load checkpoint manually from numbered directories
                    import glob
                    import os
                    shared_agent_dir = os.path.join(checkpoint_dir, 'shared_agent')
                    step_dirs = glob.glob(os.path.join(shared_agent_dir, '*'))
                    step_dirs = [d for d in step_dirs if os.path.isdir(d) and os.path.basename(d).replace('.', '').isdigit()]
                    
                    if step_dirs:
                        # Use the most recent step directory
                        latest_step_dir = max(step_dirs, key=os.path.getmtime)
                        step_num = float(os.path.basename(latest_step_dir))
                        
                        # Load checkpoint manually using Orbax
                        import orbax.checkpoint as ocp
                        try:
                            # Directly restore from the checkpoint directory (use absolute path)
                            import os
                            checkpointer = ocp.PyTreeCheckpointer()
                            abs_checkpoint_path = os.path.abspath(latest_step_dir + '/train_state')
                            train_state = checkpointer.restore(abs_checkpoint_path)
                            
                            # For SPPPO, we need to recreate the network to get apply_fn
                            # Import SPPPO network class
                            import sys
                            import jax.numpy as jnp
                            sys.path.append('/share/code/src/JaxMARL/baselines')
                            from SPPPO.spppo_ff_mpe import ActorCritic
                            import jaxmarl
                            
                            # Recreate the network (same as in SPPPO training)
                            env = jaxmarl.make("MPE_simple_sumo_v3")
                            network = ActorCritic(env.action_space(env.agents[0]).n, activation="tanh")
                            
                            # Handle different checkpoint structures
                            if hasattr(train_state, 'params'):
                                # TrainState object
                                player.params = train_state.params
                                player.apply_fn = network.apply
                            elif isinstance(train_state, dict) and 'params' in train_state:
                                # Dictionary with params
                                player.params = train_state['params']
                                player.apply_fn = network.apply
                            else:
                                # Assume train_state is the params directly
                                player.params = train_state
                                player.apply_fn = None
                            print(f"✅ Manually loaded SPPPO checkpoint from step {step_num}")
                        except Exception as e:
                            print(f"❌ Failed to manually load SPPPO checkpoint: {e}")
                            raise ValueError(f"Failed to load SPPPO checkpoint from {latest_step_dir}: {e}")
                    else:
                        raise ValueError(f"No SPPPO checkpoint found in {checkpoint_dir}")
                else:
                    train_state, _ = result
                    
                    # For SPPPO, we need to recreate the network to get apply_fn
                    from baselines.SPPPO.train import ActorCritic
                    import jaxmarl
                    
                    # Recreate the network (same as in SPPPO training)
                    env = jaxmarl.make("MPE_simple_sumo_v3")
                    network = ActorCritic(env.action_space(env.agents[0]).n, activation="tanh")
                    
                    # Handle different checkpoint structures
                    if hasattr(train_state, 'params'):
                        # TrainState object
                        player.params = train_state.params
                        player.apply_fn = network.apply
                    elif isinstance(train_state, dict) and 'params' in train_state:
                        # Dictionary with params
                        player.params = train_state['params']
                        player.apply_fn = network.apply
                    else:
                        # Assume train_state is the params directly
                        player.params = train_state
                        player.apply_fn = network.apply
                    
                    print(f"✅ Loaded SPPPO checkpoint from {checkpoint_dir}")
                
            elif player.algorithm == 'FSPPPO':
                # Load FSPPPO checkpoint
                # Extract run_id from checkpoint path
                # Path format: checkpoints/fspppo/run_xyz_seed0/main_agent/step_N
                path_parts = player.checkpoint_path.split('/')
                run_id = None
                for part in path_parts:
                    if part.startswith('run_'):
                        run_id = part
                        break
                
                if run_id is None:
                    raise ValueError(f"Could not extract run_id from path: {player.checkpoint_path}")
                
                manager = FSPPPOCheckpointManager()
                checkpoint_info = manager.get_latest_checkpoint(run_id)
                if checkpoint_info is None:
                    raise ValueError(f"No FSPPPO checkpoint found for run_id: {run_id}")
                
                # Load the checkpoint using the directory path
                checkpoint_dir = checkpoint_info["checkpoint_dir"]
                
                # Create network and abstract params structure for FSPPPO
                try:
                    from baselines.FSPPPO.train import ActorCritic
                except ImportError:
                    from FSPPPO.train import ActorCritic
                network = ActorCritic(action_dim=5, activation='tanh')
                
                # Create dummy input to initialize network params
                import jax
                import jax.numpy as jnp
                rng = jax.random.PRNGKey(0)
                dummy_obs = jnp.zeros((1, 8))  # Observation shape for simple_sumo
                abstract_params = network.init(rng, dummy_obs)
                
                # Load the checkpoint
                loaded_params = manager.load_checkpoint(checkpoint_dir, abstract_params)
                player.params = loaded_params
                player.apply_fn = network.apply
                
            print(f"✅ Successfully loaded {player.name}")
            
        except Exception as e:
            print(f"❌ Failed to load {player.name}: {e}")
            raise
    
    def get_scripted_action(self, obs, player_name: str, rng_key):
        """Get action from scripted opponent using standardized behaviors.
        
        Returns discrete action indices for simple_sumo environment:
        0: no-op, 1: up, 2: down, 3: right, 4: left
        """
        script_type = player_name.split('_')[1]  # Extract type from 'scripted_noop'
        return get_scripted_action(obs, script_type, rng_key)
    
    def run_single_episode(self, player1: TournamentPlayer, player2: TournamentPlayer, 
                          rng_key, episode_id: int) -> Dict[str, Any]:
        """Run a single episode between two players."""
        
        # Reset environment
        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = self.env.reset(reset_key)
        
        episode_rewards = {agent: 0.0 for agent in self.env.agents}
        episode_length = 0
        done = False
        
        for step in range(self.max_episode_steps):
            # Get actions from both players
            actions = {}
            
            for i, (agent, player) in enumerate(zip(self.env.agents, [player1, player2])):
                if player.player_type == 'scripted':
                    rng_key, action_key = jax.random.split(rng_key)
                    actions[agent] = self.get_scripted_action(obs[agent], player.name, action_key)
                else:
                    # Checkpoint player
                    if player.params is None:
                        self.load_checkpoint_player(player)
                    
                    # Get action from neural network
                    rng_key, action_key = jax.random.split(rng_key)
                    network_output = player.apply_fn(player.params, obs[agent])
                    
                    if isinstance(network_output, tuple):
                        # Handle different network output formats
                        pi, value = network_output
                        
                        # Check if pi is a distribution object (SPPPO case)
                        if hasattr(pi, 'sample'):
                            # It's a distrax.Categorical distribution - sample directly
                            actions[agent] = pi.sample(seed=action_key)
                        else:
                            # It's raw logits - use categorical sampling
                            actions[agent] = jax.random.categorical(action_key, pi)
                    else:
                        # Single output - assume it's logits
                        actions[agent] = jax.random.categorical(action_key, network_output)
            
            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = self.env.step(step_key, state, actions)
            
            # Accumulate rewards
            for agent in self.env.agents:
                episode_rewards[agent] += rewards[agent]
            
            episode_length += 1
            
            # Check if episode is done
            if dones["__all__"]:
                done = True
                break
        
        # Determine winner
        agent_names = list(self.env.agents)
        player1_reward = episode_rewards[agent_names[0]]
        player2_reward = episode_rewards[agent_names[1]]
        
        if player1_reward > player2_reward:
            winner = player1.name
            outcome = 1  # Player 1 wins
        elif player2_reward > player1_reward:
            winner = player2.name
            outcome = -1  # Player 2 wins
        else:
            winner = "draw"
            outcome = 0  # Draw
        
        return {
            'episode_id': episode_id,
            'player1': player1.name,
            'player2': player2.name,
            'player1_reward': float(player1_reward),
            'player2_reward': float(player2_reward),
            'winner': winner,
            'outcome': outcome,
            'episode_length': episode_length,
            'completed': done
        }
    
    def run_match(self, match: TournamentMatch, rng_key) -> List[Dict[str, Any]]:
        """Run a complete match between two players (symmetrical)."""
        print(f"🥊 Running match: {match.player1.name} vs {match.player2.name}")
        print(f"   Episodes: {match.total_episodes} ({match.episodes_per_side} per side)")
        
        match_results = []
        episode_id = 0
        
        # First half: Player1 as agent_0, Player2 as agent_1
        print(f"   Side 1: {match.player1.name} (green) vs {match.player2.name} (red)")
        for i in range(match.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            result = self.run_single_episode(match.player1, match.player2, episode_key, episode_id)
            result['side'] = 1
            result['player1_color'] = 'green'
            result['player2_color'] = 'red'
            match_results.append(result)
            episode_id += 1
            
            if (i + 1) % 10 == 0:
                print(f"     Completed {i + 1}/{match.episodes_per_side} episodes")
        
        # Second half: Player2 as agent_0, Player1 as agent_1 (swap positions)
        print(f"   Side 2: {match.player2.name} (green) vs {match.player1.name} (red)")
        for i in range(match.episodes_per_side):
            rng_key, episode_key = jax.random.split(rng_key)
            result = self.run_single_episode(match.player2, match.player1, episode_key, episode_id)
            # Swap the results to maintain consistent player naming
            result['player1'], result['player2'] = result['player2'], result['player1']
            result['player1_reward'], result['player2_reward'] = result['player2_reward'], result['player1_reward']
            result['outcome'] = -result['outcome']  # Flip outcome since positions swapped
            if result['winner'] == match.player2.name:
                result['winner'] = match.player1.name
            elif result['winner'] == match.player1.name:
                result['winner'] = match.player2.name
            
            result['side'] = 2
            result['player1_color'] = 'red'
            result['player2_color'] = 'green'
            match_results.append(result)
            episode_id += 1
            
            if (i + 1) % 10 == 0:
                print(f"     Completed {i + 1}/{match.episodes_per_side} episodes")
        
        # Calculate match statistics
        player1_wins = sum(1 for r in match_results if r['winner'] == match.player1.name)
        player2_wins = sum(1 for r in match_results if r['winner'] == match.player2.name)
        draws = sum(1 for r in match_results if r['winner'] == 'draw')
        
        print(f"   Results: {match.player1.name}: {player1_wins}, {match.player2.name}: {player2_wins}, Draws: {draws}")
        
        return match_results
    
    def setup_tournament(self, selected_players: Optional[List[str]] = None, latest_only: bool = False):
        """Set up the tournament with all players and matches."""
        
        # Discover all available players
        checkpoint_players = self.discover_checkpoint_players(latest_only=latest_only)
        scripted_players = self.create_scripted_players()
        all_players = checkpoint_players + scripted_players
        
        # Filter players if specific ones were requested
        if selected_players:
            selected_set = set(selected_players)
            if 'ippo' in selected_set:
                selected_set.update([p.name for p in checkpoint_players if 'IPPO' in p.name])
            if 'spppo' in selected_set:
                selected_set.update([p.name for p in checkpoint_players if 'SPPPO' in p.name])
            if 'fspppo' in selected_set:
                selected_set.update([p.name for p in checkpoint_players if 'FSPPPO' in p.name])
            if 'scripted' in selected_set:
                selected_set.update([p.name for p in scripted_players])
            
            self.players = [p for p in all_players if p.name in selected_set]
        else:
            self.players = all_players
        
        print(f"\n🏟️  Tournament Setup:")
        print(f"   Total players: {len(self.players)}")
        
        # Create all possible matches (round-robin)
        self.matches = []
        for player1, player2 in itertools.combinations(self.players, 2):
            match = TournamentMatch(player1, player2, self.episodes_per_side)
            self.matches.append(match)
        
        total_episodes = len(self.matches) * self.episodes_per_matchup
        print(f"   Total matches: {len(self.matches)}")
        print(f"   Total episodes: {total_episodes}")
        print(f"   Estimated duration: {total_episodes * 2 / 60:.1f} minutes")
    
    def run_tournament(self, rng_key):
        """Run the complete tournament."""
        print(f"\n🚀 Starting Tournament!")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        for i, match in enumerate(self.matches):
            print(f"\n📊 Match {i+1}/{len(self.matches)}")
            rng_key, match_key = jax.random.split(rng_key)
            match_results = self.run_match(match, match_key)
            self.results.extend(match_results)
            
            # Save intermediate results every 10 matches
            if (i + 1) % 10 == 0:
                self.save_results(intermediate=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🏆 Tournament Complete!")
        print(f"   Duration: {duration/60:.1f} minutes")
        print(f"   Total episodes: {len(self.results)}")
        
        # Save final results
        self.save_results(intermediate=False)
        self.generate_summary()
    
    def save_results(self, intermediate: bool = False):
        """Save tournament results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_intermediate" if intermediate else ""
        filename = f"tournament_results_{timestamp}{suffix}.csv"
        filepath = self.output_dir / filename
        
        if not self.results:
            print("⚠️  No results to save")
            return
        
        fieldnames = [
            'episode_id', 'player1', 'player2', 'player1_reward', 'player2_reward',
            'winner', 'outcome', 'episode_length', 'completed', 'side',
            'player1_color', 'player2_color'
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"💾 Results saved to: {filepath}")
    
    def generate_summary(self):
        """Generate tournament summary statistics."""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"tournament_summary_{timestamp}.txt"
        
        # Calculate win rates for each player
        player_stats = {}
        for player in self.players:
            player_stats[player.name] = {
                'wins': 0, 'losses': 0, 'draws': 0, 'total_reward': 0.0, 'episodes': 0
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
            f.write("🏆 TOURNAMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Tournament Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Episodes per matchup: {self.episodes_per_matchup}\n")
            f.write(f"Total players: {len(self.players)}\n")
            f.write(f"Total matches: {len(self.matches)}\n")
            f.write(f"Total episodes: {len(self.results)}\n\n")
            
            f.write("PLAYER STATISTICS\n")
            f.write("-" * 30 + "\n")
            
            # Sort players by win rate
            sorted_players = sorted(player_stats.items(), 
                                  key=lambda x: x[1]['wins'] / max(x[1]['episodes'], 1), 
                                  reverse=True)
            
            for player_name, stats in sorted_players:
                win_rate = stats['wins'] / max(stats['episodes'], 1) * 100
                avg_reward = stats['total_reward'] / max(stats['episodes'], 1)
                
                f.write(f"\n{player_name}:\n")
                f.write(f"  Win Rate: {win_rate:.1f}% ({stats['wins']}/{stats['episodes']})\n")
                f.write(f"  W/L/D: {stats['wins']}/{stats['losses']}/{stats['draws']}\n")
                f.write(f"  Avg Reward: {avg_reward:.3f}\n")
        
        print(f"📋 Summary saved to: {summary_file}")


def main():
    """Main tournament execution function."""
    parser = argparse.ArgumentParser(description="Run comprehensive tournament evaluation")
    parser.add_argument("--players", type=str, default=None,
                       help="Comma-separated list of players (e.g., 'ippo,spppo,scripted')")
    parser.add_argument("--episodes-per-matchup", type=int, default=100,
                       help="Number of episodes per matchup (default: 100)")
    parser.add_argument("--max-episode-steps", type=int, default=100,
                       help="Maximum steps per episode (default: 100)")
    parser.add_argument("--output-dir", type=str, default="tournament_results",
                       help="Output directory for results (default: tournament_results)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--latest-only", action="store_true", default=True,
                       help="Only use the most recent checkpoint from each algorithm type (default: True)")
    parser.add_argument("--all-checkpoints", action="store_true",
                       help="Use all available checkpoints (overrides --latest-only)")
    
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
    
    # Determine whether to use latest-only mode
    latest_only = args.latest_only and not args.all_checkpoints
    
    # Setup and run tournament
    evaluator.setup_tournament(selected_players, latest_only=latest_only)
    
    # Run tournament
    rng_key = jax.random.PRNGKey(args.seed)
    evaluator.run_tournament(rng_key)
    
    print("\n🎉 Tournament evaluation complete!")
    print(f"📁 Results saved in: {args.output_dir}/")


if __name__ == "__main__":
    main()
