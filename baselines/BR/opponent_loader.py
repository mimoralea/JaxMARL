"""
Opponent Loader Module for Best-Response (BR) Training

This module loads learned opponents from checkpoints and provides a unified interface
for using them in BR training. It handles the different checkpoint formats and network
architectures used by IPPO, SPPPO, and FSPPPO.
"""

import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing import Callable, Dict, Any, Tuple
from flax.training.train_state import TrainState

# Import network architectures from each algorithm
from baselines.IPPO.train import ActorCritic as IPPOActorCritic
from baselines.SPPPO.train import ActorCritic as SPPPOActorCritic  
from baselines.FSPPPO.train import ActorCritic as FSPPPOActorCritic

from .opponent_discovery import OpponentInfo


class OpponentLoader:
    """Loads learned opponents from checkpoints for BR training."""
    
    def __init__(self, env):
        """Initialize opponent loader.
        
        Args:
            env: JaxMARL environment instance
        """
        self.env = env
        self.action_space_n = env.action_space(env.agents[0]).n
        self.obs_shape = env.observation_space(env.agents[0]).shape
        
    def load_opponent(self, opponent_info: OpponentInfo) -> Tuple[Callable, Any]:
        """Load an opponent from checkpoint.
        
        Args:
            opponent_info: Information about the opponent to load
            
        Returns:
            Tuple of (apply_function, parameters) for the opponent
        """
        algorithm = opponent_info.algorithm.upper()
        
        if algorithm == "IPPO":
            return self._load_ippo_opponent(opponent_info)
        elif algorithm == "SPPPO":
            return self._load_spppo_opponent(opponent_info)
        elif algorithm == "FSPPPO":
            return self._load_fspppo_opponent(opponent_info)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _load_ippo_opponent(self, opponent_info: OpponentInfo) -> Tuple[Callable, Any]:
        """Load IPPO opponent from checkpoint."""
        # Create IPPO network
        network = IPPOActorCritic(self.action_space_n, activation="tanh")
        
        # Load checkpoint using the same logic as tournament system
        try:
            # Try PyTreeCheckpointer first
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(opponent_info.checkpoint_path)
            
            # Handle different checkpoint structures
            if 'model' in restored:
                params = restored['model']['params']
            elif 'params' in restored:
                params = restored['params']
            else:
                params = restored
                
        except Exception as e1:
            try:
                # Try loading from train_state subdirectory (IPPO/SPPPO format)
                train_state_path = os.path.join(opponent_info.checkpoint_path, 'train_state')
                if os.path.exists(train_state_path):
                    orbax_checkpointer = ocp.PyTreeCheckpointer()
                    restored = orbax_checkpointer.restore(train_state_path)
                    params = restored['params']
                else:
                    raise Exception("train_state subdirectory not found")
            except Exception as e2:
                raise Exception(f"Failed both direct PyTree ({e1}) and train_state loading ({e2})")
        
        # IPPO has two agents, we'll use the first agent's parameters
        if isinstance(params, tuple) and len(params) == 2:
            # IPPO stores params as (agent0_params, agent1_params)
            opponent_params = params[0]
        else:
            # Fallback: use params directly
            opponent_params = params
            
        print(f"Loaded IPPO opponent from {opponent_info.checkpoint_path}")
        return network.apply, opponent_params
    
    def _load_spppo_opponent(self, opponent_info: OpponentInfo) -> Tuple[Callable, Any]:
        """Load SPPPO opponent from checkpoint."""
        # Create SPPPO network
        network = SPPPOActorCritic(self.action_space_n, activation="tanh")
        
        # Load checkpoint using the same logic as tournament system
        try:
            # Try PyTreeCheckpointer first
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(opponent_info.checkpoint_path)
            
            # Handle different checkpoint structures
            if 'model' in restored:
                opponent_params = restored['model']['params']
            elif 'params' in restored:
                opponent_params = restored['params']
            else:
                opponent_params = restored
                
        except Exception as e1:
            try:
                # Try loading from train_state subdirectory (IPPO/SPPPO format)
                train_state_path = os.path.join(opponent_info.checkpoint_path, 'train_state')
                if os.path.exists(train_state_path):
                    orbax_checkpointer = ocp.PyTreeCheckpointer()
                    restored = orbax_checkpointer.restore(train_state_path)
                    opponent_params = restored['params']
                else:
                    raise Exception("train_state subdirectory not found")
            except Exception as e2:
                raise Exception(f"Failed both direct PyTree ({e1}) and train_state loading ({e2})")
            
        print(f"Loaded SPPPO opponent from {opponent_info.checkpoint_path}")
        return network.apply, opponent_params
    
    def _load_fspppo_opponent(self, opponent_info: OpponentInfo) -> Tuple[Callable, Any]:
        """Load FSPPPO opponent from checkpoint."""
        # Create FSPPPO network
        network = FSPPPOActorCritic(self.action_space_n, activation="tanh")
        
        # Load checkpoint using the same logic as tournament system
        try:
            # Try PyTreeCheckpointer first
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(opponent_info.checkpoint_path)
            
            # Handle different checkpoint structures
            if 'model' in restored:
                opponent_params = restored['model']['params']
            elif 'params' in restored:
                # FSPPPO stores params directly in the 'params' key
                # but we need to wrap them for Flax network.apply()
                opponent_params = {'params': restored['params']}
            else:
                opponent_params = restored
                
        except Exception as e1:
            try:
                # Try loading from train_state subdirectory (IPPO/SPPPO format)
                train_state_path = os.path.join(opponent_info.checkpoint_path, 'train_state')
                if os.path.exists(train_state_path):
                    orbax_checkpointer = ocp.PyTreeCheckpointer()
                    restored = orbax_checkpointer.restore(train_state_path)
                    opponent_params = restored['params']
                else:
                    raise Exception("train_state subdirectory not found")
            except Exception as e2:
                raise Exception(f"Failed both direct PyTree ({e1}) and train_state loading ({e2})")
            
        print(f"Loaded FSPPPO opponent from {opponent_info.checkpoint_path}")
        return network.apply, opponent_params
    
    def create_opponent_policy(self, opponent_info: OpponentInfo) -> Callable:
        """Create a policy function for the opponent.
        
        Args:
            opponent_info: Information about the opponent to load
            
        Returns:
            Policy function that takes (observation, rng_key) and returns action
        """
        apply_fn, params = self.load_opponent(opponent_info)
        
        def opponent_policy(obs, rng_key):
            """Opponent policy function.
            
            Args:
                obs: Observation for the opponent
                rng_key: Random key for action sampling
                
            Returns:
                Action selected by the opponent
            """
            # Ensure observation is properly shaped
            if len(obs.shape) == 1:
                # Single observation, add batch dimension
                obs_batch = obs[None, ...]
            else:
                obs_batch = obs
                
            # Get policy distribution and value
            pi, _ = apply_fn(params, obs_batch)
            
            # Sample action
            action = pi.sample(seed=rng_key)
            
            # Remove batch dimension if we added it
            if len(obs.shape) == 1:
                action = action[0]
                
            return action
        
        return opponent_policy
    
    def test_opponent_loading(self, opponent_info: OpponentInfo) -> bool:
        """Test if an opponent can be loaded successfully.
        
        Args:
            opponent_info: Information about the opponent to test
            
        Returns:
            True if loading succeeds, False otherwise
        """
        try:
            apply_fn, params = self.load_opponent(opponent_info)
            
            # Test with dummy observation
            dummy_obs = jnp.zeros(self.obs_shape)
            dummy_key = jax.random.PRNGKey(0)
            
            # Test policy execution
            pi, value = apply_fn(params, dummy_obs[None, ...])
            action = pi.sample(seed=dummy_key)
            
            print(f"✅ Successfully tested opponent: {opponent_info.name}")
            print(f"   Action shape: {action.shape}, Value shape: {value.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load opponent {opponent_info.name}: {e}")
            return False


def load_opponent_for_br(opponent_info: OpponentInfo, env) -> Callable:
    """Convenience function to load an opponent for BR training.
    
    Args:
        opponent_info: Information about the opponent to load
        env: JaxMARL environment instance
        
    Returns:
        Policy function for the opponent
    """
    loader = OpponentLoader(env)
    return loader.create_opponent_policy(opponent_info)


if __name__ == "__main__":
    # Test the opponent loader
    import jaxmarl
    from .opponent_discovery import discover_learned_opponents
    
    print("Testing opponent loader...")
    
    # Create environment
    env = jaxmarl.make("MPE_simple_sumo_v3")
    
    # Discover opponents
    opponents = discover_learned_opponents(training_seed=0, latest_only=True)
    
    if not opponents:
        print("No opponents found for testing")
    else:
        # Test loading each opponent
        loader = OpponentLoader(env)
        
        for opponent in opponents:
            print(f"\nTesting {opponent.name}...")
            success = loader.test_opponent_loading(opponent)
            
            if success:
                # Test creating policy function
                try:
                    policy = loader.create_opponent_policy(opponent)
                    
                    # Test policy with dummy data
                    dummy_obs = jnp.zeros(env.observation_space(env.agents[0]).shape)
                    dummy_key = jax.random.PRNGKey(42)
                    action = policy(dummy_obs, dummy_key)
                    
                    print(f"   Policy test successful, action: {action}")
                except Exception as e:
                    print(f"   Policy test failed: {e}")
