"""
Based on the PureJaxRL Implementation of PPO
Best-Response (BR) Agent Training - trains one agent against fixed learned opponents
"""

import time
import logging
import warnings
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import optax
import distrax
import orbax.checkpoint as ocp
import jaxmarl
from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.default_params import MAX_STEPS
import matplotlib.pyplot as plt
import wandb
import functools
from datetime import datetime
import os
from pathlib import Path
import numpy as np
from baselines import scripted_behaviors

# Configure logging levels to reduce verbose output
logging.getLogger("absl").setLevel(logging.CRITICAL)
logging.getLogger("orbax").setLevel(logging.CRITICAL)
# Suppress specific Orbax checkpoint manager warnings
warnings.filterwarnings("ignore", message=".*CheckpointManager.*asynchronous.*")
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.ERROR)
logging.getLogger("tensorstore").setLevel(logging.ERROR)

# Import shared PPO implementation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.algorithms.ppo import ActorCritic, Transition, batchify, unbatchify

# Import checkpoint manager
try:
    from .orbax_checkpoint_manager import (
        create_br_checkpoint_manager,
        BRCheckpointCallback,
    )
except ImportError:
    from baselines.BR.orbax_checkpoint_manager import (
        create_br_checkpoint_manager,
        BRCheckpointCallback,
    )

# Import baseline network architectures for loading opponent checkpoints
from baselines.IPPO.train import ActorCritic as IPPOActorCritic
from baselines.SPPPO.train import ActorCritic as SPPPOActorCritic
from baselines.FSPPPO.train import ActorCritic as FSPPPOActorCritic

# At the top of your script
# jax.config.update('jax_disable_jit', True)


# ActorCritic, Transition, batchify, and unbatchify are now imported from shared PPO module


def load_opponent_checkpoint(config, env):
    """Load opponent checkpoint from target baseline algorithm.

    Returns:
        tuple: (opponent_params, opponent_info)
            opponent_params: PyTree params or None if not found
            opponent_info: dict with numeric fields {"opponent_algo_code", "opponent_seed", "opponent_step"}
    """
    target_algo = config.get("TARGET_ALGORITHM", "ippo").lower()
    
    # Handle scripted opponents (noop, random, seek, guardian, dodge)
    if target_algo in ["noop", "random", "seek", "guardian", "dodge"]:
        print(f"   Using scripted opponent: {target_algo}")
        return "SCRIPTED_" + target_algo.upper(), {"opponent_algo_code": 0, "opponent_seed": int(config.get("SEED", 0)), "opponent_step": -1}
    
    if target_algo not in ["ippo", "spppo", "fspppo"]:
        raise ValueError(f"Unknown TARGET_ALGORITHM: {target_algo}")
    
    # Map algorithm names to their network classes
    network_classes = {
        "ippo": IPPOActorCritic,
        "spppo": SPPPOActorCritic, 
        "fspppo": FSPPPOActorCritic,
    }

    algo_code_map = {"ippo": 1, "spppo": 2, "fspppo": 3}
    
    # Look for checkpoints in the baseline directory
    checkpoint_base = f"checkpoints/{target_algo}"

    # 1) Support explicit override via OPPONENT_CHECKPOINT_PATH
    explicit_path = str(config.get("OPPONENT_CHECKPOINT_PATH", "") or "").strip()
    agent_dir_abs = None
    forced_step = None

    if explicit_path:
        explicit_path = os.path.abspath(explicit_path)
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Provided OPPONENT_CHECKPOINT_PATH does not exist: {explicit_path}")
        # If the explicit path points to a numeric step directory, its parent is the agent dir
        base_name = os.path.basename(explicit_path)
        if base_name.isdigit():
            agent_dir_abs = os.path.dirname(explicit_path)
            forced_step = int(base_name)
        else:
            # Otherwise assume it's the agent dir (e.g., .../main)
            agent_dir_abs = explicit_path

        print(f"   Using explicit opponent path: {agent_dir_abs} (step: {forced_step if forced_step is not None else 'latest'})")

    # 2) If no explicit path, find latest checkpoint across ANY seed
    if agent_dir_abs is None:
        if not os.path.exists(checkpoint_base):
            raise FileNotFoundError(f"No checkpoints found for {target_algo} at {checkpoint_base}")

        # Collect all run dirs across seeds
        all_runs = [d for d in os.listdir(checkpoint_base) if d.startswith("run_") and d.endswith(tuple([f"_seed{i}" for i in range(1000)]))]
        if not all_runs:
            raise RuntimeError(f"No run directories found for {target_algo} under {checkpoint_base}")

        # Sort lexicographically (timestamped format ensures newest last) and pick latest existing checkpoint
        all_runs.sort()
        chosen_run = None
        chosen_seed = -1
        chosen_step = -1
        chosen_agent_dir = None

        for run_dir in reversed(all_runs):
            agent_dir = os.path.join(checkpoint_base, run_dir, "main")
            if not os.path.exists(agent_dir):
                continue

            # Create manager and read latest step
            cm = ocp.CheckpointManager(
                directory=os.path.abspath(agent_dir),
                checkpointers={
                    'train_state': ocp.PyTreeCheckpointer(),
                    'metadata': ocp.StandardCheckpointer(),
                },
                options=ocp.CheckpointManagerOptions(create=False),
            )
            latest = cm.latest_step()
            if latest is not None:
                chosen_run = run_dir
                chosen_seed = int(run_dir.split("_seed")[-1]) if "_seed" in run_dir else -1
                chosen_step = int(latest)
                chosen_agent_dir = agent_dir
                break

        if chosen_agent_dir is None:
            raise RuntimeError(f"No checkpoint steps found in any run for {target_algo} under {checkpoint_base}")

        agent_dir_abs = os.path.abspath(chosen_agent_dir)
        forced_step = chosen_step
        print(f"   Selected opponent run: {chosen_run}, seed: {chosen_seed}, step: {forced_step}")

    # Create checkpoint manager for the resolved directory (for step discovery only)
    checkpoint_manager = ocp.CheckpointManager(
        directory=agent_dir_abs,
        checkpointers={'metadata': ocp.StandardCheckpointer()},
        options=ocp.CheckpointManagerOptions(create=False),
    )

    # Get step to load
    latest_step = forced_step if forced_step is not None else checkpoint_manager.latest_step()
    if latest_step is None:
        raise RuntimeError(f"No checkpoint steps found under {agent_dir_abs}")
    print(f"   Loading checkpoint from step {latest_step} at {agent_dir_abs}")
    
    # Initialize the appropriate network architecture
    network_class = network_classes[target_algo]
    obs_shape = env.observation_space(env.agents[0]).shape
    action_dim = env.action_space(env.agents[0]).n
    
    # Initialize network with same architecture as baseline
    # All baselines use the shared ActorCritic which doesn't take hidden_dims
    network = network_class(
        action_dim=action_dim,
        activation="tanh",
    )
    
    # Create dummy train state for structure
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(obs_shape)
    params = network.init(rng, init_x)
    tx = optax.adam(1e-8)  # Dummy optimizer for structure
    
    try:
        # Determine opponent seed from path if possible
        opp_seed = -1
        try:
            # agent_dir_abs: .../checkpoints/<algo>/run_..._seedX/main
            parts = agent_dir_abs.split(os.sep)
            run_seed_part = parts[-2]  # run_..._seedX
            if "_seed" in run_seed_part:
                opp_seed = int(run_seed_part.split("_seed")[-1])
        except Exception:
            opp_seed = -1

        # Build step directory path
        step_dir = os.path.join(agent_dir_abs, str(latest_step))

        # Restore strategy depends on algorithm
        if target_algo in ["ippo", "spppo"]:
            # Baselines saved via CheckpointManager with PyTreeCheckpointer under 'train_state'
            cm = ocp.CheckpointManager(
                directory=agent_dir_abs,
                checkpointers={
                    'train_state': ocp.PyTreeCheckpointer(),
                    'metadata': ocp.StandardCheckpointer(),
                },
                options=ocp.CheckpointManagerOptions(create=False),
            )
            restored = cm.restore(int(latest_step))
            if restored is None or 'train_state' not in restored:
                raise RuntimeError("Missing 'train_state' in restored checkpoint")
            ts = restored['train_state']
            # ts may be a TrainState or a dict-like
            if hasattr(ts, 'params'):
                restored_params = ts.params
            elif isinstance(ts, dict) and 'params' in ts:
                restored_params = ts['params']
            else:
                raise RuntimeError("Restored train_state has no 'params'")
        else:
            # FSPPPO saved params directly at step dir using StandardCheckpointer
            rng = jax.random.PRNGKey(0)
            dummy_params = network.init(rng, jnp.zeros(obs_shape))
            std_ckpt = ocp.StandardCheckpointer()
            restored_params = std_ckpt.restore(step_dir, dummy_params)

        print(f"   ✅ Successfully loaded opponent checkpoint from {target_algo}")
        return restored_params, {
            "opponent_algo_code": algo_code_map.get(target_algo, 0),
            "opponent_seed": int(opp_seed),
            "opponent_step": int(latest_step),
        }

    except Exception as e:
        raise RuntimeError(f"Failed to load opponent checkpoint for {target_algo} at {step_dir}: {e}")


def make_train(config, checkpoint_callback=None, opponent_params=None):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    # Calculate number of updates based on total timesteps
    # Use int() to ensure we get an integer value
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    # Recalculate effective timesteps to match configured total
    effective_timesteps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
    
    # Validate and warn if there's still a mismatch
    if effective_timesteps != int(config["TOTAL_TIMESTEPS"]):
        print(f"  Adjusted NUM_UPDATES to {config['NUM_UPDATES']} to match TOTAL_TIMESTEPS")
        print(f"  Effective timesteps: {effective_timesteps}")
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    
    # Use the standard LogWrapper for the environment
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, opponent_params):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
        )
        # Determine agent indices
        br_idx = int(config.get("BR_AGENT_INDEX", 0))
        opp_idx = int(config.get("OPPONENT_AGENT_INDEX", 1))
        
        # Handle different opponent types
        scripted_opponent_type = None
        if opponent_params is None:
            print("   Using random initialization for opponent")
            # Initialize opponent with random parameters
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            opponent_params = network.init(_rng, init_x)
        elif isinstance(opponent_params, str) and opponent_params.startswith("SCRIPTED_"):
            # Scripted opponent - no params needed
            scripted_opponent_type = opponent_params
            print(f"   Using scripted opponent: {scripted_opponent_type}")
            # Still need dummy params for structure
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            opponent_params = network.init(_rng, init_x)
        else:
            # Loaded checkpoint params
            pass
        
        # Initialize only the BR agent's parameters (like FSPPPO)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        br_agent_params = network.init(_rng, init_x)

        # Create optimizer only for BR agent (like FSPPPO)
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

        # Train state only contains BR agent parameters (like FSPPPO)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=br_agent_params,  # Only BR agent params
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(
                    last_obs, env.agents, config["NUM_ACTORS"]
                )
                # SELECT ACTIONS – BR agent vs fixed opponent (like FSPPPO)
                rng, _rng = jax.random.split(rng)
                subkeys = jax.random.split(_rng, env.num_agents)

                obs_split = jnp.split(obs_batch, env.num_agents, axis=0)
                actions = []
                values = []
                log_probs = []
                # Build actions for both agents respecting indices
                # BR agent (trainable)
                pi_br, v_br = network.apply(train_state.params, obs_split[br_idx])
                act_br = pi_br.sample(seed=subkeys[br_idx])
                lp_br = pi_br.log_prob(act_br)
                # Opponent (fixed)
                batch_size = obs_split[opp_idx].shape[0]
                # Check if we have a scripted opponent type
                if scripted_opponent_type is not None:
                    # Handle scripted opponents - need to match batch dimensions
                    behavior_name = scripted_opponent_type.replace("SCRIPTED_", "").lower()
                    
                    # Generate actions for each environment in the batch using centralized scripted behaviors
                    actions_list = []
                    current_rng = subkeys[opp_idx]
                    for i in range(batch_size):
                        obs_i = obs_split[opp_idx][i]  # Opponent observation for env i
                        rng_key_i, current_rng = jax.random.split(current_rng)
                        action_i = scripted_behaviors.get_scripted_action(obs_i, behavior_name, rng_key_i)
                        actions_list.append(action_i)
                    
                    act_opp = jnp.array(actions_list, dtype=jnp.int32)
                    v_opp = jnp.zeros_like(v_br)  # Dummy value
                    lp_opp = jnp.zeros_like(lp_br)  # Dummy log prob
                else:
                    # Use network for learned opponent
                    pi_opp, v_opp = network.apply(opponent_params, obs_split[opp_idx])
                    act_opp = pi_opp.sample(seed=subkeys[opp_idx])
                    lp_opp = pi_opp.log_prob(act_opp)  # Not used for training

                # Place actions/values/logprobs in agent order 0..num_agents-1
                for agent_idx in range(env.num_agents):
                    if agent_idx == br_idx:
                        actions.append(act_br)
                        values.append(v_br)
                        log_probs.append(lp_br)
                    else:
                        actions.append(act_opp)
                        values.append(v_opp)
                        log_probs.append(lp_opp)

                action = jnp.concatenate(actions, axis=0)
                value = jnp.concatenate(values, axis=0)
                log_prob = jnp.concatenate(log_probs, axis=0)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step,
                    env_state,
                    env_act,
                )

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(
                        reward, env.agents, config["NUM_ACTORS"]
                    ).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(
                last_obs, env.agents, config["NUM_ACTORS"]
            )
            # compute value predictions - only BR agent values are used for training
            obs_split = jnp.split(last_obs_batch, env.num_agents, axis=0)
            # BR agent (trainable)
            _, v_br = network.apply(train_state.params, obs_split[br_idx])
            # Opponent agent (fixed)
            if opponent_params is None and 'scripted_opponent_type' in locals():
                v_opp = jnp.zeros_like(v_br)
            else:
                _, v_opp = network.apply(opponent_params, obs_split[opp_idx])
            # Stack values in agent order
            last_vals = []
            for agent_idx in range(env.num_agents):
                last_vals.append(v_br if agent_idx == br_idx else v_opp)
            last_val = jnp.reshape(jnp.stack(last_vals, axis=1), (-1,))

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

                    # -------- Only train BR agent (configurable agent index) ------------
                    # Extract BR agent data only
                    sel = slice(br_idx, None, env.num_agents)
                    tb_br = jax.tree_util.tree_map(lambda x: x[sel], traj_batch)
                    gae_br = advantages[sel]
                    tgt_br = targets[sel]

                    def _loss_single(params, tb, gae, tgt):
                        pi, value = network.apply(params, tb.obs)
                        log_prob = pi.log_prob(tb.action)
                        # value loss
                        v_clipped = tb.value + (value - tb.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        v_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - tgt),
                                jnp.square(v_clipped - tgt),
                            ).mean()
                        )
                        # policy loss
                        ratio = jnp.exp(log_prob - tb.log_prob)
                        gae_n = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_pi = -jnp.minimum(
                            ratio * gae_n,
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_n,
                        ).mean()
                        entropy = pi.entropy().mean()
                        total = (
                            loss_pi
                            + config["VF_COEF"] * v_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total, (v_loss, loss_pi, entropy, ratio.mean())

                    # compute grads for BR agent only (like FSPPPO)
                    (loss_br, aux_br), grads_br = jax.value_and_grad(
                        _loss_single, has_aux=True
                    )(train_state.params, tb_br, gae_br, tgt_br)
                    
                    # Update only BR agent parameters (like FSPPPO)
                    updates, new_opt_state = train_state.tx.update(
                        grads_br, train_state.opt_state, train_state.params
                    )
                    new_params = optax.apply_updates(train_state.params, updates)
                    train_state = train_state.replace(
                        params=new_params, opt_state=new_opt_state
                    )
                    
                    # log only BR agent metrics
                    loss_info = {
                        "total_loss": loss_br,
                        "actor_loss": aux_br[1],
                        "critic_loss": aux_br[0],
                        "entropy": aux_br[2],
                        "ratio": aux_br[3],
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
            # Separate envs vs agents
            per_agent = step_average.reshape(
                (config["NUM_ENVS"], env.num_agents)
            )
            # Mean over envs
            env_average = per_agent.mean(axis=0) / env.num_agents

            rng = update_state[-1]
            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            # Add per-agent returns after blanket mean (respect indices)
            metric["br_agent_returns"] = env_average[br_idx]
            metric["opponent_returns"] = env_average[opp_idx]
            # For backward compatibility (keep indices 0/1 for plots)
            metric["player_0_returns"] = env_average[0]
            metric["player_1_returns"] = env_average[1]
            metric = {**metric, **loss_info, **r0}

            # Store update index in metrics for logging outside JIT
            metric["update_idx"] = update_idx

            # No IO callbacks inside JIT-compiled and vmapped code
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="br_ff_mpe")
def main(config):
    """Train with BR (Best-Response) then generate rollouts via eval_arena for demo."""
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["BR", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Setup checkpoint management (end-only for speed)
    checkpoint_enabled = config.get("SAVE_AT_END", True)
    if checkpoint_enabled:
        print("Checkpoint management enabled: save at end only (for speed)")

        # Generate base run ID for this training session
        import datetime

        base_run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

        # Create checkpoint managers for each seed
        checkpoint_managers = []
        checkpoint_callbacks = []

        for seed_idx in range(config["NUM_SEEDS"]):
            # Determine target algorithm from config (consistent key with rest of codebase)
            target_algorithm = config.get("TARGET_ALGORITHM", "self_play").lower()
            manager = create_br_checkpoint_manager(
                run_id=base_run_id,
                seed=seed_idx,
                target_algorithm=target_algorithm,
                max_to_keep=config.get("MAX_CHECKPOINTS_TO_KEEP", 10),
                agent_names=["main_agent"],
            )
            callback = BRCheckpointCallback(
                checkpoint_manager=manager,
                save_frequency=999999,  # Large number to disable intermediate saves
                save_at_end=config["SAVE_AT_END"],
            )
            checkpoint_managers.append(manager)
            checkpoint_callbacks.append(callback)
    else:
        print("Checkpoint management disabled")
        checkpoint_managers = None
        checkpoint_callbacks = None
        base_run_id = None

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

    # Validate and report effective total timesteps
    # Convert TOTAL_TIMESTEPS to int if it's in scientific notation
    if isinstance(config['TOTAL_TIMESTEPS'], float):
        config['TOTAL_TIMESTEPS'] = int(config['TOTAL_TIMESTEPS'])
    
    # Print training schedule validation
    print("Training schedule validation:")
    print(f"  TOTAL_TIMESTEPS (configured): {config['TOTAL_TIMESTEPS']}")
    effective_timesteps = config['NUM_ENVS'] * config['NUM_STEPS'] * config['NUM_UPDATES']
    print(f"  NUM_ENVS x NUM_STEPS x NUM_UPDATES: {config['NUM_ENVS']} x {config['NUM_STEPS']} x {config['NUM_UPDATES']} = {effective_timesteps}")
    
    # Adjust NUM_UPDATES if needed to match TOTAL_TIMESTEPS
    if effective_timesteps != config['TOTAL_TIMESTEPS']:
        config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // (config['NUM_ENVS'] * config['NUM_STEPS'])
        effective_timesteps = config['NUM_ENVS'] * config['NUM_STEPS'] * config['NUM_UPDATES']
        print(f"  ✅ Adjusted NUM_UPDATES to {config['NUM_UPDATES']} to match TOTAL_TIMESTEPS")
        print(f"  New effective timesteps: {effective_timesteps}")
    else:
        print("  ✅ Effective timesteps match configured TOTAL_TIMESTEPS")

    # Simple progress message
    print(
        f"Training with JIT enabled for {config['NUM_SEEDS']} seeds, {int(config['NUM_UPDATES'])} updates each"
    )
    print(
        f"(Total of {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} updates)"
    )

    # Load opponent checkpoint before JIT compilation
    print("\n🎯 Loading opponent checkpoint from {} baseline...".format(
        config.get("TARGET_ALGORITHM", "ippo").upper()
    ))
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    opponent_params, opponent_meta = load_opponent_checkpoint(config, env)
    
    if opponent_params is None:
        print("   Using random initialization for opponent")
    else:
        print("   Successfully loaded opponent checkpoint")
    
    # Attach opponent metadata to checkpoint callbacks so it's persisted in BR checkpoints
    if checkpoint_callbacks is not None:
        for cb in checkpoint_callbacks:
            try:
                cb.extra_metadata = opponent_meta
            except AttributeError:
                # Older callback without extra_metadata support; ignore gracefully
                pass
    
    # Create JIT-compiled training function with opponent params
    print("\nCompiling training function with JAX JIT...")
    train_fn = make_train(config, checkpoint_callback=None, opponent_params=opponent_params)
    
    # Create a wrapper that passes opponent_params to the train function
    def train_with_opponent(rng):
        return train_fn(rng, opponent_params)
    
    train_jit = jax.jit(train_with_opponent)

    # Run training across all seeds in parallel (using vmap for speed)
    print("\nRunning training (first run includes compilation time)...")
    out = jax.vmap(train_jit)(rngs)

    # Save checkpoints after parallel training completes
    if checkpoint_enabled:
        print("\nSaving final checkpoints...")
        for seed_idx in range(config["NUM_SEEDS"]):
            # Extract train_state for this seed using the same pattern as original code
            # out["runner_state"] shape: (num_seeds, 4) where 4 = (train_state, env_state, obsv, rng)
            # We want train_state (index 0) for each seed
            seed_runner_state = jax.tree_util.tree_map(
                lambda x: x[seed_idx], out["runner_state"]
            )
            train_state = seed_runner_state[
                0
            ]  # train_state is at index 0 of the tuple
            checkpoint_callbacks[seed_idx].save_final_checkpoint(
                train_state=train_state, step=config["NUM_UPDATES"]
            )
            print(f"Final checkpoint saved for seed {seed_idx}")

    print(
        f"\nTraining complete! Processed {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} total updates"
    )
    metrics = out["metrics"]

    # Extract the trained model parameters from the first seed
    # With vmap: out["runner_state"] is a tuple (train_state, env_state, obsv, rng)
    # where each element has shape (num_seeds, ...)
    # We want train_state (index 0 of tuple) from seed 0 (index 0 of batch)
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])

    # Quick 10-episode evaluation vs noop, random, and the original opponent checkpoint
    def _eval_quick(br_params, opponent_params, label: str, episodes: int = 10):
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        action_n = env.action_space(env.agents[0]).n
        net = ActorCritic(action_n, activation=config["ACTIVATION"])
        apply_fn = jax.jit(net.apply)

        def _run_episode(rng, opponent_kind):
            rng, reset_key = jax.random.split(rng)
            obs, state = env.reset(reset_key)
            total = {a: 0.0 for a in env.agents}
            steps = 0
            while steps < MAX_STEPS:
                rng, a_key, b_key = jax.random.split(rng, 3)
                # BR agent (green/agent 0)
                pi0, _ = apply_fn(br_params, obs[env.agents[0]])
                a0 = pi0.sample(seed=a_key)

                # Opponent (red/agent 1)
                if opponent_kind == "noop":
                    a1 = jnp.array(0, dtype=jnp.int32)
                elif opponent_kind == "random":
                    a1 = jax.random.randint(b_key, (), 0, action_n, dtype=jnp.int32)
                elif opponent_kind == "checkpoint":
                    # Support both learned and scripted opponents for the 'checkpoint' slot.
                    if isinstance(opponent_params, str) and opponent_params.startswith("SCRIPTED_"):
                        behavior_name = opponent_params.replace("SCRIPTED_", "").lower()
                        a1 = scripted_behaviors.get_scripted_action(
                            obs[env.agents[1]], behavior_name, b_key
                        )
                    elif opponent_params is None:
                        # Fallback to random if not available
                        a1 = jax.random.randint(b_key, (), 0, action_n, dtype=jnp.int32)
                    else:
                        pi1, _ = apply_fn(opponent_params, obs[env.agents[1]])
                        a1 = pi1.sample(seed=b_key)

                acts = {env.agents[0]: a0, env.agents[1]: a1}
                rng, step_key = jax.random.split(rng)
                obs, state, rew, done, _ = env.step(step_key, state, acts)
                for a in env.agents:
                    total[a] += float(rew[a])
                steps += 1
                if done.get("__all__", False):
                    break
            return total

        rng = jax.random.PRNGKey(int(time.time() * 1000) % 100000)
        results = {"noop": [], "random": [], "checkpoint": []}
        for opponent_kind in ["noop", "random", "checkpoint"]:
            for ep in range(episodes):
                rng, ep_key = jax.random.split(rng)
                totals = _run_episode(ep_key, opponent_kind)
                results[opponent_kind].append(totals)

        def _summarize(kind):
            br = np.mean([r[env.agents[0]] for r in results[kind]])
            opp = np.mean([r[env.agents[1]] for r in results[kind]])
            wins = sum(1 for r in results[kind] if r[env.agents[0]] > r[env.agents[1]])
            draws = sum(1 for r in results[kind] if r[env.agents[0]] == r[env.agents[1]])
            return br, opp, wins, draws

    
        for kind in ["noop", "random", "checkpoint"]:
            br_avg, opp_avg, wins, draws = _summarize(kind)
            print(f"[EVAL:{label}] vs {kind:10s} -> BR avg {br_avg:.3f}, OP avg {opp_avg:.3f}, wins {wins}/{episodes}, draws {draws}/{episodes}")

    print("\nRunning quick 10-episode evaluation vs noop, random, and opponent checkpoint...")
    _eval_quick(train_state.params, opponent_params, label=config.get("TARGET_ALGORITHM", "unknown"), episodes=10)

    # Generate rollouts for different opponent types (include the original checkpoint opponent)
    opponent_types = ["self_play", "noop", "random", "checkpoint"]
    print("\nGenerating rollout animations against different opponents (including checkpoint opponent)...")

    # Use current time-based seeds to ensure different starting positions each run
    base_seed = int(time.time() * 1000) % 100000

    # Get run_id for structured rollout folder organization
    if checkpoint_enabled:
        rollout_run_id = base_run_id
    else:
        import datetime

        rollout_run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    for i, opponent_type in enumerate(opponent_types):
        print(f"\nGenerating rollout against {opponent_type} opponent...")
        # Use different seed for each opponent type by adding the index
        rollout_seed = base_seed + i
        get_rollout(
            train_state,
            config,
            opponent_type=opponent_type,
            seed=rollout_seed,
            run_id=rollout_run_id,
            training_seed=0,
            opponent_params=opponent_params if opponent_type == "checkpoint" else None,
        )

    # Get the environment name to check if it's a zero-sum game
    env_name = config["ENV_NAME"].lower()

    plt.figure(figsize=(10, 6))

    # For simple_sumo, we need to look at other metrics since returns are all zeros
    if "sumo" in env_name:
        # Create a figure with multiple subplots to show different metrics
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Get per-agent returns from a rollout
        ax = axs[0]
        # Convert JAX arrays to NumPy and keep the seeds dimension to compute statistics
        player_0 = np.asarray(
            out["metrics"]["player_0_returns"]
        )  # shape (seeds, updates)
        player_1 = np.asarray(
            out["metrics"]["player_1_returns"]
        )  # shape (seeds, updates)

        # Average over seeds and compute standard deviation for error bars
        mean_0, std_0 = player_0.mean(0), player_0.std(0)
        mean_1, std_1 = player_1.mean(0), player_1.std(0)

        x = np.arange(mean_0.shape[0])  # Updates on the x-axis

        # Plot mean curve with shaded ±1 std region
        ax.plot(x, mean_0, label="Player 0", color="green", linewidth=2)
        ax.fill_between(
            x, mean_0 - std_0, mean_0 + std_0, color="green", alpha=0.2
        )

        ax.plot(x, mean_1, label="Player 1", color="red", linewidth=2)
        ax.fill_between(
            x, mean_1 - std_1, mean_1 + std_1, color="red", alpha=0.2
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
        plt.savefig(f"br_ff_{config['ENV_NAME']}.png")


def get_rollout(
    train_state,
    config,
    opponent_type="self_play",
    seed=None,
    run_id=None,
    training_seed=0,
    opponent_params=None,
):
    """Generate a rollout of the environment for visualization

    Args:
        train_state: The trained agent's parameters
        config: Configuration dictionary
        opponent_type: Type of opponent ('self_play', 'noop', or 'random')
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
    # BR now only trains one agent, so train_state.params is just the BR agent params
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
            if opponent_type == "self_play":  # Both use trained policy
                agent_obs = obs[second_agent].flatten()
                pi, _ = network.apply(network_params, agent_obs)
                action = pi.sample(seed=key_a)
                actions[second_agent] = action

            elif opponent_type == "noop":  # Opponent does nothing
                # 0 = NOOP in discrete action space
                actions[second_agent] = jnp.array(0, dtype=jnp.int32)

            elif (
                opponent_type == "random"
            ):  # Opponent takes random actions
                # Random action from 0-4 (NOOP, LEFT, RIGHT, DOWN, UP)
                key_rand, key = jax.random.split(key)
                random_action = jax.random.randint(key_rand, (), 0, 5)
                actions[second_agent] = jnp.array(
                    random_action, dtype=jnp.int32
                )
            elif opponent_type == "checkpoint":
                # Use the exact checkpoint parameters loaded for the BR opponent.
                # If a scripted type was used for training, respect that here too.
                if isinstance(opponent_params, str) and opponent_params.startswith("SCRIPTED_"):
                    behavior_name = opponent_params.replace("SCRIPTED_", "").lower()
                    action = scripted_behaviors.get_scripted_action(
                        obs[second_agent].flatten(), behavior_name, key_a
                    )
                    actions[second_agent] = jnp.array(action, dtype=jnp.int32)
                elif opponent_params is None:
                    # Fallback: treat as random if not provided to avoid crash
                    key_rand, key = jax.random.split(key)
                    random_action = jax.random.randint(key_rand, (), 0, 5)
                    actions[second_agent] = jnp.array(
                        random_action, dtype=jnp.int32
                    )
                else:
                    agent_obs = obs[second_agent].flatten()
                    pi, _ = network.apply(opponent_params, agent_obs)
                    action = pi.sample(seed=key_a)
                    actions[second_agent] = action

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

    # Create structured folder path: rollouts/br/run_id/seed_X/
    if run_id is None:
        import datetime

        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    rollouts_base_dir = "rollouts"
    algorithm_dir = os.path.join(rollouts_base_dir, "br")
    run_dir = os.path.join(algorithm_dir, run_id)
    seed_dir = os.path.join(run_dir, f"seed_{training_seed}")
    os.makedirs(seed_dir, exist_ok=True)

    viz = MPEVisualizer(env, state_seq, reward_seq=reward_seq)
    gif_filename = f"br_ff_{config['ENV_NAME']}_{opponent_type}.gif"
    gif_path = os.path.join(seed_dir, gif_filename)
    viz.animate(save_fname=gif_path, view=False, loop=False)
    print(f"Animation saved to {gif_path}")

    return state_seq, reward_seq


if __name__ == "__main__":
    main()
