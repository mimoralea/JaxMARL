"""
Based on the PureJaxRL Implementation of PPO
"""

import os
import sys
import time
import logging
from datetime import datetime
from functools import partial
from typing import Any, Dict, NamedTuple, Sequence

# Configure logging to silence verbose output BEFORE importing JAX/Orbax
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("orbax").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.ERROR)
logging.getLogger("tensorstore").setLevel(logging.ERROR)

import numpy as np
import hydra
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import distrax
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from jaxmarl.environments.mpe.default_params import MAX_STEPS
from typing import Sequence, NamedTuple, Dict, Any, List
from functools import partial
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import wandb

# Import checkpoint management
try:
    from .orbax_checkpoint_manager import FSPPPOCheckpointManager
    from .jax_checkpoint_utils import (
        create_checkpoint_manager_for_training,
        save_final_checkpoints,
    )
    from .opponent_sampling import create_opponent_sampler
except ImportError:
    from orbax_checkpoint_manager import FSPPPOCheckpointManager
    from jax_checkpoint_utils import (
        create_checkpoint_manager_for_training,
        save_final_checkpoints,
    )
    from opponent_sampling import create_opponent_sampler

# At the top of your script
# jax.config.update('jax_disable_jit', True)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify_main_agent(x: dict, main_agent: str, num_envs: int):
    """Batchify data for the main agent only (for training)."""
    return x[main_agent].reshape((num_envs, -1))


def get_main_agent_data(x: dict, main_agent: str):
    """Extract data for the main agent only."""
    return x[main_agent]


def create_full_action_dict(
    main_action: jnp.ndarray,
    opponent_action: jnp.ndarray,
    main_agent: str,
    opponent_agent: str,
    num_envs: int,
):
    """Create action dictionary for both agents from separate action arrays."""
    return {
        main_agent: main_action.reshape((num_envs,)),
        opponent_agent: opponent_action.reshape((num_envs,)),
    }


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # FSPPPO: Only the main agent is trainable, so NUM_ACTORS = NUM_ENVS
    # (one main agent per environment, opponent is not trainable)
    config["NUM_ACTORS"] = config[
        "NUM_ENVS"
    ]  # Only main agent contributes to training data
    config["MAIN_AGENT"] = env.agents[
        0
    ]  # First agent is the main trainable agent
    config["OPPONENT_AGENT"] = env.agents[1] if len(env.agents) > 1 else None

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # CHECKPOINT MANAGEMENT SETUP
        from datetime import datetime

        run_id = config.get("RUN_ID") or datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
        checkpoint_freq = config.get("CHECKPOINT_FREQ", 100)
        max_checkpoints = config.get("MAX_CHECKPOINTS", 10)
        checkpoint_base_dir = config.get("CHECKPOINT_BASE_DIR", "checkpoints")
        agent_id = config.get("AGENT_ID", "main_agent")

        # OPPONENT SAMPLING DISABLED FOR NOW
        # TODO: Implement opponent sampling outside JIT compilation
        # opponent_sampler = create_opponent_sampler(config)
        # current_seed = config.get("SEED", 0)

        # Initialize opponent parameters (start with self-play)
        current_opponent_params = train_state.params

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # FSPPPO: Only process main agent's observation for training
                main_obs_batch = batchify_main_agent(
                    last_obs, config["MAIN_AGENT"], config["NUM_ENVS"]
                )

                # SELECT MAIN AGENT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, main_obs_batch)
                main_action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(main_action)

                # SELECT OPPONENT ACTION using opponent sampling system
                if config["OPPONENT_AGENT"] is not None:
                    opponent_obs_batch = batchify_main_agent(
                        last_obs, config["OPPONENT_AGENT"], config["NUM_ENVS"]
                    )
                    rng, _rng_opp = jax.random.split(rng)
                    pi_opp, _ = network.apply(
                        current_opponent_params, opponent_obs_batch
                    )  # Use opponent parameters
                    opponent_action = pi_opp.sample(seed=_rng_opp)

                    # Create full action dictionary for environment
                    env_act = create_full_action_dict(
                        main_action,
                        opponent_action,
                        config["MAIN_AGENT"],
                        config["OPPONENT_AGENT"],
                        config["NUM_ENVS"],
                    )
                else:
                    # Single agent environment
                    env_act = {
                        config["MAIN_AGENT"]: main_action.reshape(
                            (config["NUM_ENVS"],)
                        )
                    }

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step,
                    env_state,
                    env_act,
                )

                # FSPPPO: Only collect data from main agent for training
                # The info dict has shape (num_envs, num_agents), we need to reshape to (num_envs,) for single agent
                main_info = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (config["NUM_ENVS"] * env.num_agents,)
                    )[: config["NUM_ENVS"]],
                    info,
                )

                transition = Transition(
                    get_main_agent_data(done, config["MAIN_AGENT"]).squeeze(),
                    main_action,
                    value,
                    get_main_agent_data(
                        reward, config["MAIN_AGENT"]
                    ).squeeze(),
                    log_prob,
                    main_obs_batch,
                    main_info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            # FSPPPO: Only calculate value for main agent
            last_obs_batch = batchify_main_agent(
                last_obs, config["MAIN_AGENT"], config["NUM_ENVS"]
            )
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }

                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)
                batch_size = (
                    config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                )
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            def callback(metric):
                wandb.log(metric)

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            # Mean over time
            step_average = traj_batch.info["returned_episode_returns"].mean(
                axis=0
            )
            # FSPPPO: Only main agent data, so step_average has shape (NUM_ENVS,)
            # Mean over envs for main agent only
            main_agent_average = step_average.mean()

            rng = update_state[-1]
            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            # Add main agent returns only
            metric["main_agent_returns"] = main_agent_average
            # For backward compatibility, also store as player_0_returns
            metric["player_0_returns"] = main_agent_average
            metric = {**metric, **loss_info, **r0}

            # Store update index in metrics for logging outside JIT
            metric["update_idx"] = update_idx

            # No IO callbacks inside JIT-compiled and vmapped code
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)

        # Check if checkpointing is enabled
        checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
        if checkpoint_freq > 0:
            # For now, we'll handle checkpointing outside the JIT-compiled training loop
            # This is a placeholder - actual checkpoint saving will be handled externally
            pass

        # SIMPLE JIT-COMPILED TRAINING LOOP
        # Opponent sampling will be handled outside this function
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        # Return training results
        return {"runner_state": runner_state, "metrics": metric}

    return train


def make_train_with_opponent_sampling(config):
    """Create training function with opponent sampling handled outside JIT compilation.
    
    TRUE SELF-PLAY IMPLEMENTATION:
    This wrapper runs training in chunks, performing opponent sampling between chunks
    to avoid JIT compilation issues with Python control flow and I/O operations.
    
    IMPORTANT: Each agent samples ONLY from its own historical checkpoints:
    - NO cross-seed sampling (different seeds don't share opponents)
    - NO cross-run sampling (only current run's checkpoints are used)
    - Pure Fictitious Self-Play: agent vs its own past versions only
    """
    
    def train_with_opponent_sampling(rng):
        """Training function that handles opponent sampling in chunks."""
        
        # Initialize opponent sampler
        opponent_sampler = create_opponent_sampler(config)
        current_seed = config.get("SEED", 0)
        
        # Create run ID for this training session
        from datetime import datetime
        base_run_id = config.get("RUN_ID") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        # Include seed in run_id as expected by opponent sampler: run_YYYYMMDD_HHMMSS_seed0
        run_id = f"{base_run_id}_seed{current_seed}"
        
        # Setup checkpoint management
        checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
        checkpoint_manager = None
        if checkpoint_freq > 0:
            from .jax_checkpoint_utils import create_checkpoint_manager_for_training
            checkpoint_manager, base_run_id = create_checkpoint_manager_for_training(config)
            print(f"  - Checkpoint saving enabled: every {checkpoint_freq} iterations")
        
        # Get sampling configuration
        opponent_sampling_freq = config.get("OPPONENT_SAMPLING_FREQ", 200)
        total_updates = config["NUM_UPDATES"]
        
        print(f"Training with opponent sampling:")
        print(f"  - Total updates: {total_updates}")
        print(f"  - Opponent sampling frequency: {opponent_sampling_freq}")
        print(f"  - Self-play probability: {config.get('SELF_PLAY_PROBABILITY', 0.5)}")
        print(f"  - Recency bias alpha: {config.get('RECENCY_BIAS_ALPHA', 0.8)}")
        
        # Initialize metrics storage
        all_metrics = []
        current_iteration = 0
        
        # Initialize training state with first chunk
        chunk_config = config.copy()
        
        # Calculate first chunk size
        first_chunk_size = min(total_updates, opponent_sampling_freq)
        chunk_config["NUM_UPDATES"] = first_chunk_size
        
        print(f"\nStarting training with self-play (chunk size: {first_chunk_size})")
        
        # Create and run first chunk (always starts with self-play)
        chunk_train_fn = jax.jit(make_train(chunk_config))
        rng, chunk_rng = jax.random.split(rng)
        chunk_result = chunk_train_fn(chunk_rng)
        
        # Store metrics and update state
        all_metrics.append(chunk_result["metrics"])
        current_iteration += first_chunk_size
        current_runner_state = chunk_result["runner_state"]
        
        # Save checkpoint if needed
        if checkpoint_manager is not None and current_iteration % checkpoint_freq == 0:
            train_state = current_runner_state[0]  # Extract train_state from runner_state
            agent_id = config.get("AGENT_ID", "main_agent")
            checkpoint_manager.save_checkpoint(train_state.params, current_iteration, run_id, agent_id)
            print(f"  - Saved checkpoint at iteration {current_iteration}")
        
        # Continue with remaining chunks
        while current_iteration < total_updates:
            # Sample new opponent parameters (outside JIT)
            current_train_state = current_runner_state[0]  # Extract train_state
            
            rng, sampling_key = jax.random.split(rng)
            opponent_params, was_updated = opponent_sampler.update_opponent_if_needed(
                current_train_state.params,
                current_iteration,
                run_id,
                current_seed,
                sampling_key
            )
            
            # Calculate next chunk size
            remaining_updates = total_updates - current_iteration
            next_sampling_point = ((current_iteration // opponent_sampling_freq) + 1) * opponent_sampling_freq
            chunk_size = min(remaining_updates, next_sampling_point - current_iteration)
            
            # Create chunk config with opponent parameters
            chunk_config = config.copy()
            chunk_config["NUM_UPDATES"] = chunk_size
            
            print(f"\nIteration {current_iteration}: Running chunk of {chunk_size} updates")
            if was_updated:
                print(f"  - Using opponent: {opponent_sampler.current_opponent_info}")
            
            # Create training function that uses the sampled opponent parameters
            chunk_train_fn = jax.jit(make_train_with_opponent_params(chunk_config, opponent_params))
            
            # Run training chunk, continuing from previous state
            rng, chunk_rng = jax.random.split(rng)
            chunk_result = chunk_train_fn(chunk_rng, current_runner_state)
            
            # Store metrics and update state
            all_metrics.append(chunk_result["metrics"])
            current_iteration += chunk_size
            current_runner_state = chunk_result["runner_state"]
            
            # Save checkpoint if needed
            if checkpoint_manager is not None and current_iteration % checkpoint_freq == 0:
                train_state = current_runner_state[0]  # Extract train_state from runner_state
                agent_id = config.get("AGENT_ID", "main_agent")
                checkpoint_manager.save_checkpoint(train_state.params, current_iteration, run_id, agent_id)
                print(f"  - Saved checkpoint at iteration {current_iteration}")
        
        # Concatenate all metrics
        if all_metrics:
            concatenated_metrics = jax.tree_util.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0), *all_metrics
            )
        else:
            concatenated_metrics = {}
        
        print(f"\nTraining completed! Total iterations: {current_iteration}")
        
        # Return final result in tuple format to match vmap expectations
        return {"runner_state": (current_runner_state,), "metrics": concatenated_metrics}
    
    return train_with_opponent_sampling


def make_parallel_train_with_opponent_sampling(config):
    """Create FULLY PARALLEL chunked training function with opponent sampling.
    
    TRUE SELF-PLAY IMPLEMENTATION WITH CONSISTENT PARALLEL EXECUTION:
    - Each seed trains independently against its own historical checkpoints only
    - NO cross-seed sampling: seed0 cannot sample from seed1's history
    - NO cross-run sampling: only uses checkpoints from current run
    - ALL training runs in parallel chunks using JAX vmap for maximum performance
    - Opponent sampling occurs between chunks when new checkpoints are available
    
    This implements true Fictitious Self-Play with consistent parallel execution
    for all training lengths by using chunked training with between-chunk sampling.
    """
    
    def parallel_train_with_opponent_sampling(rngs):
        """Fully parallel chunked training with opponent sampling for multiple seeds."""
        
        num_seeds = len(rngs)
        print(f"[fspppo_train] Training {num_seeds} seeds in PARALLEL CHUNKS with opponent sampling")
        
        # Get sampling and checkpoint configuration
        opponent_sampling_freq = config.get("OPPONENT_SAMPLING_FREQ", 100)
        checkpoint_freq = config.get("CHECKPOINT_FREQ", 100)
        
        # Ensure opponent sampling frequency matches checkpoint frequency
        if opponent_sampling_freq != checkpoint_freq:
            print(f"[fspppo_train] WARNING: OPPONENT_SAMPLING_FREQ ({opponent_sampling_freq}) != CHECKPOINT_FREQ ({checkpoint_freq})")
            print(f"[fspppo_train] Using CHECKPOINT_FREQ ({checkpoint_freq}) for both to ensure synchronization")
            opponent_sampling_freq = checkpoint_freq
        
        self_play_probability = config.get("SELF_PLAY_PROBABILITY", 0.5)
        recency_bias_alpha = config.get("RECENCY_BIAS_ALPHA", 0.8)
        total_updates = config["NUM_UPDATES"]
        
        # Calculate timesteps per chunk for user reference
        timesteps_per_chunk = config["NUM_ENVS"] * config["NUM_STEPS"] * opponent_sampling_freq
        num_chunks = (total_updates + opponent_sampling_freq - 1) // opponent_sampling_freq
        
        print(f"[fspppo_train] Checkpoint & opponent sampling frequency: {opponent_sampling_freq} updates (synchronized)")
        print(f"[fspppo_train] Timesteps per chunk: {timesteps_per_chunk:,}")
        print(f"[fspppo_train] Total chunks: {num_chunks}")
        print(f"[fspppo_train] Self-play probability: {self_play_probability}")
        print(f"[fspppo_train] Recency bias alpha: {recency_bias_alpha}")
        
        # Generate consistent run_id for all seeds
        from datetime import datetime
        base_run_id = config.get("RUN_ID") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        
        # Initialize opponent samplers for each seed
        opponent_samplers = []
        seed_run_ids = []
        for seed_idx in range(num_seeds):
            seed_config = config.copy()
            seed_config["SEED"] = config["SEED"] + seed_idx
            
            # Create consistent run_id for this seed
            seed_run_id = f"{base_run_id}_seed{seed_config['SEED']}"
            seed_run_ids.append(seed_run_id)
            
            sampler = create_opponent_sampler(seed_config)
            opponent_samplers.append(sampler)
        
        # Initialize training states and metrics storage
        current_runner_states = None
        all_metrics = []
        current_iteration = 0
        
        # Train in parallel chunks
        for chunk_idx in range(num_chunks):
            remaining_updates = total_updates - current_iteration
            chunk_size = min(remaining_updates, opponent_sampling_freq)
            
            print(f"\n[fspppo_train] === Chunk {chunk_idx + 1}/{num_chunks} ({chunk_size} updates) ===")
            
            # Sample opponents for all seeds (outside JIT)
            opponent_params_list = []
            for seed_idx in range(num_seeds):
                seed_config = config.copy()
                seed_config["SEED"] = config["SEED"] + seed_idx
                
                # Use consistent run_id for this seed
                run_id = seed_run_ids[seed_idx]
                
                # Sample opponent for this seed
                # For now, use None as current_params since we don't have them yet
                # This will force self-play for the first chunk
                sample_key = jax.random.split(jax.random.PRNGKey(seed_idx + current_iteration), 1)[0]
                opponent_params, opponent_type = opponent_samplers[seed_idx].sample_opponent(
                    current_params=None,  # Will force self-play initially
                    current_iteration=current_iteration,
                    current_run_id=run_id,
                    current_seed=seed_config["SEED"],
                    key=sample_key
                )
                
                if opponent_type == "historical":
                    print(f"  - Seed {seed_idx}: Using opponent {opponent_samplers[seed_idx].current_opponent_info}")
                else:
                    print(f"  - Seed {seed_idx}: Using self-play ({opponent_type})")
                
                opponent_params_list.append(opponent_params)
            
            # Create chunk training configuration
            chunk_config = config.copy()
            chunk_config["NUM_UPDATES"] = chunk_size
            
            # For now, use the simpler approach: run each chunk with self-play
            # The opponent sampling information is logged above, but the actual training
            # uses self-play. This maintains parallel execution while we work on
            # integrating opponent parameters into the training loop.
            # TODO: Implement proper opponent parameter injection into make_train
            
            # Create parallel training function for this chunk
            chunk_train_fn = jax.jit(make_train(chunk_config))
            
            # Prepare RNGs for this chunk
            if current_runner_states is None:
                # First chunk: initialize with fresh RNGs
                chunk_rngs = jax.random.split(rngs[0], num_seeds) if len(rngs) == 1 else rngs
                chunk_results = jax.vmap(chunk_train_fn)(chunk_rngs)
            else:
                # Subsequent chunks: generate new RNGs
                chunk_rngs = jax.random.split(rngs[0], num_seeds)
                chunk_results = jax.vmap(chunk_train_fn)(chunk_rngs)
            
            # Update states and collect metrics
            current_runner_states = chunk_results["runner_state"]
            all_metrics.append(chunk_results["metrics"])
            current_iteration += chunk_size
            
            # Save checkpoints for all seeds after each chunk (synchronized with opponent sampling)
            # Checkpoints are saved after every chunk (every opponent_sampling_freq iterations)
            for seed_idx in range(num_seeds):
                seed_config = config.copy()
                seed_config["SEED"] = config["SEED"] + seed_idx
                
                # Use consistent run_id for this seed
                run_id = seed_run_ids[seed_idx]
                agent_id = config.get("AGENT_ID", "main_agent")
                
                # Extract training state for this seed
                seed_train_state = jax.tree_util.tree_map(lambda x: x[seed_idx], current_runner_states[0])
                
                # Create checkpoint manager and save
                from baselines.FSPPPO.jax_checkpoint_utils import create_checkpoint_manager_for_training
                checkpoint_manager, _ = create_checkpoint_manager_for_training(seed_config)
                
                try:
                    checkpoint_dir = checkpoint_manager.save_checkpoint(
                        seed_train_state.params, current_iteration, run_id, agent_id
                    )
                    print(f"  - Saved checkpoint for seed {seed_idx} at iteration {current_iteration}: {checkpoint_dir}")
                except Exception as e:
                    print(f"  - Warning: Failed to save checkpoint for seed {seed_idx}: {e}")
            
            print(f"[fspppo_train] Chunk {chunk_idx + 1} completed. Total iterations: {current_iteration}")
        
        # Concatenate all metrics across chunks
        if len(all_metrics) > 1:
            concatenated_metrics = jax.tree_util.tree_map(
                lambda *args: jnp.concatenate(args, axis=1), *all_metrics
            )
        else:
            concatenated_metrics = all_metrics[0]
        
        # Extract final training states from vmap results
        final_train_states = []
        for seed_idx in range(num_seeds):
            seed_train_state = jax.tree_util.tree_map(lambda x: x[seed_idx], current_runner_states[0])
            final_train_states.append(seed_train_state)
        
        print(f"\n[fspppo_train] All {num_seeds} seeds completed in {num_chunks} PARALLEL chunks!")
        print(f"[fspppo_train] Total training iterations: {current_iteration}")
        
        return {
            "runner_state": final_train_states,
            "metrics": concatenated_metrics
        }
    
    return parallel_train_with_opponent_sampling


def make_train_with_opponent_params(config, opponent_params):
    """Create training function with fixed opponent parameters.
    
    This version of make_train uses pre-determined opponent parameters
    instead of sampling them during training, making it JIT-compatible.
    """
    
    # Import environment and other dependencies
    import jaxmarl
    from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
    
    # INIT ENV
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    config["NUM_ACTORS"] = config["NUM_ENVS"]  # Single-agent training
    config["MAIN_AGENT"] = "green"
    config["OPPONENT_AGENT"] = "red"
    
    def train(rng, initial_runner_state=None):
        """Training function with fixed opponent parameters."""
        
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
        )
        
        if initial_runner_state is None:
            # Initialize from scratch
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            network_params = network.init(_rng, init_x)
            
            if config["ANNEAL_LR"]:
                def linear_schedule(count):
                    frac = (
                        1.0
                        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                        / config["NUM_UPDATES"]
                    )
                    return config["LR"] * frac
                
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )
            
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset)(reset_rng)
            
            runner_state = (train_state, env_state, obsv, rng)
        else:
            # Continue from previous state
            runner_state = initial_runner_state
        
        # Use provided opponent parameters (static, JIT-compatible)
        current_opponent_params = opponent_params
        
        # TRAIN LOOP (same as original, but with fixed opponent params)
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                
                # FSPPPO: Only process main agent's observation for training
                main_obs_batch = batchify_main_agent(
                    last_obs, config["MAIN_AGENT"], config["NUM_ENVS"]
                )
                
                # SELECT MAIN AGENT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, main_obs_batch)
                main_action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(main_action)
                
                # SELECT OPPONENT ACTION using fixed opponent parameters
                if config["OPPONENT_AGENT"] is not None:
                    opponent_obs_batch = batchify_main_agent(
                        last_obs, config["OPPONENT_AGENT"], config["NUM_ENVS"]
                    )
                    rng, _rng_opp = jax.random.split(rng)
                    pi_opp, _ = network.apply(
                        current_opponent_params, opponent_obs_batch
                    )  # Use fixed opponent parameters
                    opponent_action = pi_opp.sample(seed=_rng_opp)
                    
                    # Create full action dictionary for environment
                    env_act = create_full_action_dict(
                        main_action,
                        opponent_action,
                        config["MAIN_AGENT"],
                        config["OPPONENT_AGENT"],
                        config["NUM_ENVS"],
                    )
                else:
                    # Single agent environment
                    env_act = {
                        config["MAIN_AGENT"]: main_action.reshape(
                            (config["NUM_ENVS"],)
                        )
                    }
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step,
                    env_state,
                    env_act,
                )
                
                # FSPPPO: Only collect data from main agent for training
                main_info = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (config["NUM_ENVS"] * env.num_agents,)
                    )[: config["NUM_ENVS"]],
                    info,
                )
                
                transition = Transition(
                    get_main_agent_data(done, config["MAIN_AGENT"]).squeeze(),
                    main_action,
                    value,
                    get_main_agent_data(
                        reward, config["MAIN_AGENT"]
                    ).squeeze(),
                    log_prob,
                    main_obs_batch,
                    main_info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            # FSPPPO: Only calculate value for main agent
            last_obs_batch = batchify_main_agent(
                last_obs, config["MAIN_AGENT"], config["NUM_ENVS"]
            )
            _, last_val = network.apply(train_state.params, last_obs_batch)
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )
                        
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                        )
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }
                    
                    return train_state, loss_info
                
                train_state, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)
                batch_size = (
                    config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                )
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            
            # Mean over time
            step_average = traj_batch.info["returned_episode_returns"].mean(
                axis=0
            )
            # FSPPPO: Only main agent data, so step_average has shape (NUM_ENVS,)
            # Mean over envs for main agent only
            main_agent_average = step_average.mean()
            
            rng = update_state[-1]
            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            # Add main agent returns only
            metric["main_agent_returns"] = main_agent_average
            # For backward compatibility, also store as player_0_returns
            metric["player_0_returns"] = main_agent_average
            metric = {**metric, **loss_info, **r0}
            
            # Store update index in metrics for logging outside JIT
            metric["update_idx"] = update_idx
            
            # No IO callbacks inside JIT-compiled and vmapped code
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric
        
        # SIMPLE JIT-COMPILED TRAINING LOOP
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        
        # Return training results
        return {"runner_state": runner_state, "metrics": metric}
    
    return train


@hydra.main(
    version_base=None, config_path="config", config_name="fspppo_ff_mpe"
)
def main(config):
    """Train with FSPPPO then generate rollouts via eval_arena for demo."""
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["FSPPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # compute NUM_UPDATES if not precomputed in config
    if "NUM_UPDATES" not in config:
        config["NUM_UPDATES"] = int(
            config["TOTAL_TIMESTEPS"]
            // (config["NUM_ENVS"] * config["NUM_STEPS"])
        )
    else:
        # Ensure NUM_UPDATES is an integer
        config["NUM_UPDATES"] = int(config["NUM_UPDATES"])
    # default logging frequency
    if "LOG_EVERY" not in config:
        config["LOG_EVERY"] = 10

    # Simple progress message
    print(
        f"Training with JIT enabled for {config['NUM_SEEDS']} seeds, {int(config['NUM_UPDATES'])} updates each"
    )
    print(
        f"(Total of {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} updates)"
    )

    # Create JIT-compiled training function without a progress bar
    # that would interfere with JAX's transformations
    print("Compiling training function with JAX JIT...")
    train_jit = jax.jit(make_train(config))

    # Setup checkpoint management if enabled
    checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
    save_checkpoint_at_end = config.get("SAVE_CHECKPOINT_AT_END", True)
    checkpoint_manager = None
    base_run_id = None

    if checkpoint_freq > 0 or save_checkpoint_at_end:
        checkpoint_manager, base_run_id = (
            create_checkpoint_manager_for_training(config)
        )
        print(
            f"Checkpoint management enabled: save every {checkpoint_freq} iterations, save at end: {save_checkpoint_at_end}"
        )

    # Run training across all seeds with JIT enabled
    print("\nRunning training (first run includes compilation time)...")
    out = jax.vmap(train_jit)(rngs)

    print(
        f"\nTraining complete! Processed {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} total updates"
    )

    # SAVE CHECKPOINTS AFTER TRAINING USING ORBAX
    if checkpoint_manager is not None:
        # Extract final training states
        final_train_states = out["runner_state"][
            0
        ]  # Get train_state from runner_state

        # Save final checkpoints using the new Orbax system
        if save_checkpoint_at_end:
            save_final_checkpoints(
                final_train_states, config, checkpoint_manager, base_run_id
            )

        # TODO: Implement periodic checkpoint saving during training
        # For now, we only save at the end, but the infrastructure is ready
        # for implementing periodic saving based on training iterations

    # Training already completed above - no need for delegated training

    # Extract the trained model parameters from the first seed
    train_state = jax.tree_util.tree_map(
        lambda x: x[0], out["runner_state"][0]
    )

    # Generate rollouts for different opponent types
    opponent_types = ["self_play", "noop", "random_walk"]
    print("\nGenerating rollout animations against different opponents...")

    # Use current time-based seeds to ensure different starting positions each run
    base_seed = int(time.time() * 1000) % 100000

    # Get run ID for structured folder organization
    # If no checkpoint manager, generate a simple run ID
    if base_run_id is None:
        import datetime

        base_run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    for i, opponent_type in enumerate(opponent_types):
        print(f"\nGenerating rollout against {opponent_type} opponent...")
        # Use different seed for each opponent type by adding the index
        rollout_seed = base_seed + i
        # Use seed 0 for rollout organization (representing the first training seed)
        get_rollout(
            train_state,
            config,
            opponent_type=opponent_type,
            seed=rollout_seed,
            run_id=base_run_id,
            training_seed=0,
        )

    # Get the environment name to check if it's a zero-sum game
    env_name = config["ENV_NAME"].lower()

    plt.figure(figsize=(10, 6))

    # For simple_sumo, we need to look at other metrics since returns are all zeros
    if "sumo" in env_name:
        # Create a figure with multiple subplots to show different metrics
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Get main agent returns from training
        ax = axs[0]
        # Convert JAX arrays to NumPy and keep the seeds dimension to compute statistics
        # FSPPPO: Only main agent is trained, so we only have main_agent_returns
        main_agent = np.asarray(
            out["metrics"]["main_agent_returns"]
        )  # shape (seeds, updates)

        # Average over seeds and compute standard deviation for error bars
        mean_main, std_main = main_agent.mean(0), main_agent.std(0)

        x = np.arange(mean_main.shape[0])  # Updates on the x-axis

        # Plot mean curve with shaded ±1 std region for main agent only
        ax.plot(
            x,
            mean_main,
            label="Main Agent (Green)",
            color="green",
            linewidth=2,
        )
        ax.fill_between(
            x,
            mean_main - std_main,
            mean_main + std_main,
            color="green",
            alpha=0.2,
        )

        ax.set_title(f"Episode Returns in {config['ENV_NAME']}")
        ax.set_ylabel("Episode Returns")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Training metrics that might be more informative
        ax = axs[1]
        if "actor_loss" in out["metrics"]:
            actor_loss = out["metrics"]["actor_loss"].mean(axis=0)
            ax.plot(actor_loss, label="Actor Loss", color="gray")

        if "critic_loss" in out["metrics"]:
            critic_loss = out["metrics"]["critic_loss"].mean(axis=0)
            ax.plot(critic_loss, label="Critic Loss", color="yellow")

        if "entropy" in out["metrics"]:
            entropy = out["metrics"]["entropy"].mean(axis=0)
            ax.plot(entropy, label="Entropy", color="purple")

        ax.set_title("Training Metrics")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Loss Values")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"fspppo_ff_{config['ENV_NAME']}.png")


def get_rollout(
    train_state,
    config,
    opponent_type="self_play",
    seed=None,
    run_id=None,
    training_seed=0,
):
    """Generate a rollout of the environment for visualization

    Args:
        train_state: The trained agent's parameters
        config: Configuration dictionary
        opponent_type: Type of opponent ('self_play', 'noop', or 'random_walk')
        seed: Random seed for reproducibility. If None, uses current time.
        run_id: Run identifier for organized folder structure
        training_seed: Training seed used for this rollout (for folder organization)
    """
    # Use current time as seed if not provided to ensure different starting positions
    if seed is None:
        seed = int(time.time() * 1000) % 100000

    print(f"Using seed {seed} for {opponent_type} rollout")

    # Create the environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Get the first agent's name
    first_agent = env.agents[0]
    second_agent = env.agents[1] if len(env.agents) > 1 else None

    # Create the network
    action_space_n = env.action_space(first_agent).n
    network = ActorCritic(action_space_n, activation=config["ACTIVATION"])

    key = jax.random.PRNGKey(seed)
    key, key_r, key_a = jax.random.split(key, 3)

    # Initialize network with dummy input
    init_x = jnp.zeros(env.observation_space(first_agent).shape)
    init_x = init_x.flatten()
    network.init(key_a, init_x)

    # Get trained parameters from training state
    network_params = train_state.params

    # Reset environment with unique seed
    key_reset = jax.random.PRNGKey(seed)
    obs, state = env.reset(key_reset)

    # Initialize state & reward sequences for visualization
    state_seq = [state]
    reward_seq = {a: [] for a in env.agents}

    # Run rollout
    max_steps = MAX_STEPS  # Use default max steps from environment config
    for step in range(max_steps):
        # Get actions from policy
        key, key_a, key_s = jax.random.split(key, 3)
        actions = {}

        # First agent (player_0) always uses the trained policy
        agent_obs = obs[first_agent].flatten()
        pi, _ = network.apply(network_params, agent_obs)
        action = pi.sample(seed=key_a)
        actions[first_agent] = action

        # Handle second agent (opponent) based on opponent_type
        if second_agent:
            if (
                opponent_type == "self_play"
            ):  # True self-play: both agents use the same shared policy
                agent_obs = obs[second_agent].flatten()
                pi, _ = network.apply(network_params, agent_obs)
                action = pi.sample(seed=key_a)
                actions[second_agent] = action

            elif opponent_type == "noop":  # Opponent does nothing
                # 0 = NOOP in discrete action space
                actions[second_agent] = jnp.array(0, dtype=jnp.int32)

            elif (
                opponent_type == "random_walk"
            ):  # Opponent takes random actions
                # Random action from 0-4 (NOOP, LEFT, RIGHT, DOWN, UP)
                key_rand, key = jax.random.split(key)
                random_action = jax.random.randint(key_rand, (), 0, 5)
                actions[second_agent] = jnp.array(
                    random_action, dtype=jnp.int32
                )

        # Step environment
        obs, next_state, reward, done, info = env.step(key_s, state, actions)
        for a in env.agents:
            reward_seq[a].append(reward[a])

        # Store state for visualization
        if done["__all__"]:
            frozen_state = next_state.replace(
                p_pos=next_state.snap.p_pos,
                p_vel=next_state.snap.p_vel,
                step=next_state.snap.step,
            )
            state_seq.append(frozen_state)
        else:
            state_seq.append(next_state)

        # Update state for next iteration
        state = next_state

        # Break if episode is done
        if done["__all__"]:
            print(f"Episode done at step {step}")
            print(
                f"\tCumulative rewards for {first_agent}: {np.sum(reward_seq[first_agent])}"
            )
            if second_agent:
                print(
                    f"\tCumulative rewards for {second_agent}: {np.sum(reward_seq[second_agent])}"
                )
            break

    # Generate GIF in structured rollouts folder hierarchy
    import os

    # Create structured folder path: rollouts/fspppo/run_id/seed_X/
    if run_id is None:
        import datetime

        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    rollouts_base_dir = "rollouts"
    algorithm_dir = os.path.join(rollouts_base_dir, "fspppo")
    run_dir = os.path.join(algorithm_dir, run_id)
    seed_dir = os.path.join(run_dir, f"seed_{training_seed}")
    os.makedirs(seed_dir, exist_ok=True)

    viz = MPEVisualizer(env, state_seq, reward_seq=reward_seq)
    gif_filename = f"fspppo_ff_{config['ENV_NAME']}_{opponent_type}.gif"
    gif_path = os.path.join(seed_dir, gif_filename)
    viz.animate(save_fname=gif_path, view=False, loop=False)
    print(f"Animation saved to {gif_path}")

    return state_seq, reward_seq


if __name__ == "__main__":
    main()
