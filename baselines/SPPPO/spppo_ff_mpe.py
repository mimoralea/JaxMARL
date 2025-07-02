"""
Based on the PureJaxRL Implementation of PPO
"""

import os
import time
import numpy as np
import hydra
import jaxmarl
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
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
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

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
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
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
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
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

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

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                wandb.log(
                    metric
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            # Mean over time
            step_average = traj_batch.info["returned_episode_returns"].mean(axis=0)
            # Separate envs vs agents
            per_agent = step_average.reshape((config["NUM_ENVS"], env.num_agents))
            # Mean over envs
            env_average = per_agent.mean(axis=0) / env.num_agents

            rng = update_state[-1]
            r0 = {"ratio0": loss_info["ratio"][0,0].mean()}
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            # Add per-agent returns after blanket mean
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


@hydra.main(version_base=None, config_path="config", config_name="spppo_ff_mpe")
def main(config):
    """Train with SPPPO then generate rollouts via eval_arena for demo."""
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["SPPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # compute NUM_UPDATES if not precomputed in config
    if "NUM_UPDATES" not in config:
        config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"]))
    else:
        # Ensure NUM_UPDATES is an integer
        config["NUM_UPDATES"] = int(config["NUM_UPDATES"])
    # default logging frequency
    if "LOG_EVERY" not in config:
        config["LOG_EVERY"] = 10

    # Simple progress message
    print(f"Training with JIT enabled for {config['NUM_SEEDS']} seeds, {int(config['NUM_UPDATES'])} updates each")
    print(f"(Total of {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} updates)")

    # Create JIT-compiled training function without a progress bar
    # that would interfere with JAX's transformations
    print("Compiling training function with JAX JIT...")
    train_jit = jax.jit(make_train(config))

    # Run training across all seeds with JIT enabled
    print("\nRunning training (first run includes compilation time)...")
    out = jax.vmap(train_jit)(rngs)

    print(f"\nTraining complete! Processed {int(config['NUM_UPDATES'] * config['NUM_SEEDS'])} total updates")
    metrics = out["metrics"]

    # ---- Delegated training ----
    from baselines.SPPPO.train_spppo import train_spppo
    train_state, metrics = train_spppo(config)

    # Extract the trained model parameters from the first seed
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])

    # Generate rollouts for different opponent types
    opponent_types = ["self_play", "noop", "random_walk"]
    print("\nGenerating rollout animations against different opponents...")

    # Use current time-based seeds to ensure different starting positions each run
    base_seed = int(time.time() * 1000) % 100000

    for i, opponent_type in enumerate(opponent_types):
        print(f"\nGenerating rollout against {opponent_type} opponent...")
        # Use different seed for each opponent type by adding the index
        rollout_seed = base_seed + i
        get_rollout(train_state, config, opponent_type=opponent_type, seed=rollout_seed)

    # Get the environment name to check if it's a zero-sum game
    env_name = config['ENV_NAME'].lower()

    plt.figure(figsize=(10, 6))

    # For simple_sumo, we need to look at other metrics since returns are all zeros
    if 'sumo' in env_name:
        # Create a figure with multiple subplots to show different metrics
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Get per-agent returns from a rollout
        ax = axs[0]
        # Convert JAX arrays to NumPy and keep the seeds dimension to compute statistics
        player_0 = np.asarray(out["metrics"]["player_0_returns"])  # shape (seeds, updates)
        player_1 = np.asarray(out["metrics"]["player_1_returns"])  # shape (seeds, updates)

        # Average over seeds and compute standard deviation for error bars
        mean_0, std_0 = player_0.mean(0), player_0.std(0)
        mean_1, std_1 = player_1.mean(0), player_1.std(0)

        x = np.arange(mean_0.shape[0])   # Updates on the x-axis

        # Plot mean curve with shaded ±1 std region
        ax.plot(x, mean_0, label="Player 0", color='green', linewidth=2)
        ax.fill_between(x, mean_0 - std_0, mean_0 + std_0, color='green', alpha=0.2)

        ax.plot(x, mean_1, label="Player 1", color='red', linewidth=2)
        ax.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='red', alpha=0.2)

        ax.set_title(f"Episode Returns in {config['ENV_NAME']}")
        ax.set_ylabel("Episode Returns")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Training metrics that might be more informative
        ax = axs[1]
        if 'actor_loss' in out['metrics']:
            actor_loss = out['metrics']['actor_loss'].mean(axis=0)
            ax.plot(actor_loss, label="Actor Loss", color='gray')

        if 'critic_loss' in out['metrics']:
            critic_loss = out['metrics']['critic_loss'].mean(axis=0)
            ax.plot(critic_loss, label="Critic Loss", color='yellow')

        if 'entropy' in out['metrics']:
            entropy = out['metrics']['entropy'].mean(axis=0)
            ax.plot(entropy, label="Entropy", color='purple')

        ax.set_title("Training Metrics")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Loss Values")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"spppo_ff_{config['ENV_NAME']}.png")


def get_rollout(train_state, config, opponent_type="self_play", seed=None):
    """Generate a rollout of the environment for visualization

    Args:
        train_state: The trained agent's parameters
        config: Configuration dictionary
        opponent_type: Type of opponent ('self_play', 'noop', or 'random_walk')
        seed: Random seed for reproducibility. If None, uses current time.
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
            if opponent_type == "self_play":  # True self-play: both agents use the same shared policy
                agent_obs = obs[second_agent].flatten()
                pi, _ = network.apply(network_params, agent_obs)
                action = pi.sample(seed=key_a)
                actions[second_agent] = action

            elif opponent_type == "noop":  # Opponent does nothing
                # 0 = NOOP in discrete action space
                actions[second_agent] = jnp.array(0, dtype=jnp.int32)

            elif opponent_type == "random_walk":  # Opponent takes random actions
                # Random action from 0-4 (NOOP, LEFT, RIGHT, DOWN, UP)
                key_rand, key = jax.random.split(key)
                random_action = jax.random.randint(key_rand, (), 0, 5)
                actions[second_agent] = jnp.array(random_action, dtype=jnp.int32)

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
            print(f"\tCumulative rewards for player 0: {np.sum(reward_seq['player_0'])}")
            print(f"\tCumulative rewards for player 1: {np.sum(reward_seq['player_1'])}")
            break

    # Generate GIF
    viz = MPEVisualizer(env, state_seq, reward_seq=reward_seq)
    gif_filename = f"spppo_ff_{config['ENV_NAME']}_{opponent_type}.gif"
    viz.animate(save_fname=gif_filename, view=False, loop=False)
    print(f"Animation saved to {gif_filename}")

    return state_seq, reward_seq

if __name__ == "__main__":
    main()
