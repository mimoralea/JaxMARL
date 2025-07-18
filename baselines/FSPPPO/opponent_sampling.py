#!/usr/bin/env python3
"""
Opponent sampling system for Fictitious Self-Play PPO (FSPPPO).

This module implements recency-biased opponent sampling that combines:
1. Self-play (using current agent weights)
2. Historical opponent sampling (from saved checkpoints with recency bias)

The recency bias uses a Beta distribution to interpolate between:
- α = 0.0: Only oldest checkpoint
- α = 0.5: Uniform distribution over all checkpoints  
- α = 1.0: Only newest checkpoint
"""

import os
import glob
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import beta
import numpy as np

try:
    from .orbax_checkpoint_manager import FSPPPOCheckpointManager
except ImportError:
    from orbax_checkpoint_manager import FSPPPOCheckpointManager


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    path: str
    update_step: int
    seed: int
    run_id: str
    agent_id: str = "main_agent"
    
    def __post_init__(self):
        """Extract metadata from checkpoint path if not provided."""
        if self.update_step is None or self.seed is None or self.run_id is None:
            self._parse_path()
    
    def _parse_path(self):
        """Parse checkpoint path to extract metadata."""
        # Expected path: checkpoints/fspppo/run_xyz_seed0/main_agent/step_000123/
        path_parts = Path(self.path).parts
        
        # Extract step number
        step_dir = [p for p in path_parts if p.startswith('step_')]
        if step_dir:
            self.update_step = int(step_dir[0].split('_')[1])
        
        # Extract seed and run_id
        run_seed_dir = [p for p in path_parts if p.startswith('run_') and 'seed' in p]
        if run_seed_dir:
            parts = run_seed_dir[0].split('_seed')
            self.run_id = parts[0]
            self.seed = int(parts[1])
        
        # Extract agent_id
        agent_dirs = [p for p in path_parts if 'agent' in p]
        if agent_dirs:
            self.agent_id = agent_dirs[0]


class OpponentSampler:
    """
    Handles opponent sampling for Fictitious Self-Play PPO.
    
    Combines self-play (current weights) with historical opponent sampling
    using recency-biased selection from saved checkpoints.
    """
    
    def __init__(self, 
                 checkpoint_base_dir: str,
                 self_play_probability: float = 0.5,
                 recency_bias_alpha: float = 0.8,
                 opponent_sampling_freq: int = 200,
                 max_checkpoint_age: Optional[int] = None):
        """
        Initialize opponent sampler.
        
        Args:
            checkpoint_base_dir: Base directory for checkpoints
            self_play_probability: Probability of using current agent weights [0, 1]
            recency_bias_alpha: Recency bias parameter [0, 1]
                - 0.0: Only oldest checkpoint
                - 0.5: Uniform distribution
                - 1.0: Only newest checkpoint
            opponent_sampling_freq: Sample new opponent every N training iterations
            max_checkpoint_age: Maximum age of checkpoints to consider (None = no limit)
        """
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.self_play_probability = self_play_probability
        self.recency_bias_alpha = recency_bias_alpha
        self.opponent_sampling_freq = opponent_sampling_freq
        self.max_checkpoint_age = max_checkpoint_age
        
        # Current opponent state
        self.current_opponent_params = None
        self.current_opponent_info = None
        self.last_sampling_iteration = 0
        
        # Checkpoint manager for loading
        self.checkpoint_manager = None
    
    def discover_available_checkpoints(self, 
                                     current_run_id: str,
                                     current_seed: int) -> List[CheckpointInfo]:
        """
        Discover available opponent checkpoints from the current run only.
        
        Args:
            current_run_id: Current training run ID (e.g., "run_20250718_145204")
            current_seed: Current training seed
            
        Returns:
            List of available checkpoint information from current run
        """
        checkpoints = []
        
        # Search pattern: checkpoints/fspppo/{current_run_id}_seed{current_seed}/main_agent/step_*/
        # This ensures we only sample from the current run's checkpoints
        run_seed_dir = f"{current_run_id}_seed{current_seed}"
        search_pattern = str(self.checkpoint_base_dir / "fspppo" / run_seed_dir / "main_agent" / "step_*")
        
        for checkpoint_path in glob.glob(search_pattern):
            if os.path.isdir(checkpoint_path):
                try:
                    checkpoint_info = CheckpointInfo(path=checkpoint_path,
                                                   update_step=None,
                                                   seed=None,
                                                   run_id=None)
                    
                    # Verify this is from the current run and seed
                    if (checkpoint_info.run_id == current_run_id and 
                        checkpoint_info.seed == current_seed):
                        checkpoints.append(checkpoint_info)
                        
                except Exception as e:
                    print(f"Warning: Could not parse checkpoint path {checkpoint_path}: {e}")
                    continue
        
        # Sort by update step (oldest first)
        checkpoints.sort(key=lambda x: x.update_step)
        
        # Apply age limit if specified
        if self.max_checkpoint_age is not None and checkpoints:
            newest_step = checkpoints[-1].update_step
            min_step = newest_step - self.max_checkpoint_age
            checkpoints = [c for c in checkpoints if c.update_step >= min_step]
        
        return checkpoints
    
    def _map_alpha_to_beta_params(self, alpha: float) -> Tuple[float, float]:
        """Map recency bias alpha to Beta distribution parameters."""
        if alpha == 0.5:
            return 1.0, 1.0  # Uniform
        elif alpha < 0.5:
            # Bias toward older checkpoints
            a = 2 * alpha + 0.1
            b = 2.0
            return a, b
        else:
            # Bias toward newer checkpoints
            a = 2.0
            b = 2 * (1 - alpha) + 0.1
            return a, b
    
    def _calculate_recency_weights(self, 
                                 num_checkpoints: int, 
                                 alpha: float) -> jnp.ndarray:
        """Calculate sampling weights using Beta distribution."""
        if num_checkpoints == 1:
            return jnp.array([1.0])
        
        a, b = self._map_alpha_to_beta_params(alpha)
        
        # Generate positions in [0, 1] (avoiding exact 0 and 1)
        positions = jnp.linspace(0.01, 0.99, num_checkpoints)
        
        # Calculate Beta PDF weights
        weights = beta.pdf(positions, a, b)
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def sample_opponent_checkpoint(self, 
                                 available_checkpoints: List[CheckpointInfo],
                                 key: jax.random.PRNGKey) -> Optional[CheckpointInfo]:
        """
        Sample an opponent checkpoint using recency bias.
        
        Args:
            available_checkpoints: List of available checkpoints
            key: JAX random key
            
        Returns:
            Selected checkpoint info, or None if no checkpoints available
        """
        if not available_checkpoints:
            return None
        
        if len(available_checkpoints) == 1:
            return available_checkpoints[0]
        
        # Calculate recency-biased weights
        weights = self._calculate_recency_weights(len(available_checkpoints), 
                                                self.recency_bias_alpha)
        
        # Sample checkpoint
        selected_idx = jrandom.choice(key, len(available_checkpoints), p=weights)
        return available_checkpoints[selected_idx]
    
    def load_opponent_parameters(self, checkpoint_info: CheckpointInfo) -> Dict[str, Any]:
        """
        Load opponent parameters from checkpoint.
        
        Args:
            checkpoint_info: Checkpoint to load
            
        Returns:
            Loaded opponent parameters
        """
        if self.checkpoint_manager is None:
            # Initialize checkpoint manager if needed
            self.checkpoint_manager = FSPPPOCheckpointManager(
                base_dir=str(self.checkpoint_base_dir)
            )
        
        try:
            # Load checkpoint
            loaded_state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=checkpoint_info.path
            )
            
            if loaded_state is None:
                raise ValueError(f"Failed to load checkpoint from {checkpoint_info.path}")
            
            return loaded_state['params']
            
        except Exception as e:
            print(f"Error loading opponent checkpoint {checkpoint_info.path}: {e}")
            return None
    
    def should_sample_new_opponent(self, current_iteration: int) -> bool:
        """
        Check if we should sample a new opponent.
        
        Args:
            current_iteration: Current training iteration
            
        Returns:
            True if we should sample a new opponent
        """
        return (current_iteration - self.last_sampling_iteration) >= self.opponent_sampling_freq
    
    def sample_opponent(self, 
                       current_params: Dict[str, Any],
                       current_iteration: int,
                       current_run_id: str,
                       current_seed: int,
                       key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], str]:
        """
        Sample opponent parameters (self-play or historical).
        
        Args:
            current_params: Current agent parameters
            current_iteration: Current training iteration
            current_run_id: Current training run ID
            current_seed: Current training seed
            key: JAX random key
            
        Returns:
            (opponent_params, opponent_type) where opponent_type is "self_play" or "historical"
        """
        key1, key2 = jrandom.split(key)
        
        # Decide between self-play and historical opponent
        use_self_play = jrandom.uniform(key1) < self.self_play_probability
        
        if use_self_play:
            # Use current agent weights (self-play)
            self.current_opponent_params = current_params
            self.current_opponent_info = f"self_play_iter_{current_iteration}"
            return current_params, "self_play"
        
        else:
            # Sample from historical checkpoints
            available_checkpoints = self.discover_available_checkpoints(current_run_id, current_seed)
            
            if not available_checkpoints:
                # Fallback to self-play if no checkpoints available
                print(f"Warning: No historical checkpoints found, falling back to self-play")
                self.current_opponent_params = current_params
                self.current_opponent_info = f"self_play_fallback_iter_{current_iteration}"
                return current_params, "self_play"
            
            # Sample historical opponent
            selected_checkpoint = self.sample_opponent_checkpoint(available_checkpoints, key2)
            
            if selected_checkpoint is None:
                # Fallback to self-play
                print(f"Warning: Failed to sample historical checkpoint, falling back to self-play")
                self.current_opponent_params = current_params
                self.current_opponent_info = f"self_play_fallback_iter_{current_iteration}"
                return current_params, "self_play"
            
            # Load opponent parameters
            opponent_params = self.load_opponent_parameters(selected_checkpoint)
            
            if opponent_params is None:
                # Fallback to self-play
                print(f"Warning: Failed to load opponent parameters, falling back to self-play")
                self.current_opponent_params = current_params
                self.current_opponent_info = f"self_play_fallback_iter_{current_iteration}"
                return current_params, "self_play"
            
            # Success - using historical opponent
            self.current_opponent_params = opponent_params
            self.current_opponent_info = f"historical_step_{selected_checkpoint.update_step}"
            return opponent_params, "historical"
    
    def update_opponent_if_needed(self,
                                current_params: Dict[str, Any],
                                current_iteration: int,
                                current_run_id: str,
                                current_seed: int,
                                key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], bool]:
        """
        Update opponent parameters if sampling frequency reached.
        
        Args:
            current_params: Current agent parameters
            current_iteration: Current training iteration
            current_run_id: Current training run ID
            current_seed: Current training seed
            key: JAX random key
            
        Returns:
            (opponent_params, was_updated) where was_updated indicates if opponent changed
        """
        if self.should_sample_new_opponent(current_iteration):
            # Sample new opponent
            opponent_params, opponent_type = self.sample_opponent(
                current_params, current_iteration, current_run_id, current_seed, key
            )
            
            self.last_sampling_iteration = current_iteration
            
            print(f"Iteration {current_iteration}: Sampled new opponent - {opponent_type} "
                  f"({self.current_opponent_info})")
            
            return opponent_params, True
        
        else:
            # Keep current opponent
            if self.current_opponent_params is None:
                # First time - initialize with self-play
                self.current_opponent_params = current_params
                self.current_opponent_info = f"initial_self_play_iter_{current_iteration}"
                return current_params, True
            
            return self.current_opponent_params, False
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get current sampling configuration and state."""
        return {
            "self_play_probability": self.self_play_probability,
            "recency_bias_alpha": self.recency_bias_alpha,
            "opponent_sampling_freq": self.opponent_sampling_freq,
            "max_checkpoint_age": self.max_checkpoint_age,
            "current_opponent_info": self.current_opponent_info,
            "last_sampling_iteration": self.last_sampling_iteration
        }


def create_opponent_sampler(config: Dict[str, Any]) -> OpponentSampler:
    """
    Create opponent sampler from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OpponentSampler instance
    """
    return OpponentSampler(
        checkpoint_base_dir=config.get("CHECKPOINT_BASE_DIR", "checkpoints"),
        self_play_probability=config.get("SELF_PLAY_PROBABILITY", 0.5),
        recency_bias_alpha=config.get("RECENCY_BIAS_ALPHA", 0.8),
        opponent_sampling_freq=config.get("OPPONENT_SAMPLING_FREQ", 200),
        max_checkpoint_age=config.get("MAX_CHECKPOINT_AGE", None)
    )
