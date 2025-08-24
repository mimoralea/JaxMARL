"""
Base PPO Implementation for Multi-Agent Reinforcement Learning

This module provides the core PPO training logic that can be shared across
different multi-agent algorithms (IPPO, SPPPO, FSPPPO).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax
import numpy as np
from typing import NamedTuple, Sequence, Dict, Any, Callable, Tuple
from functools import partial
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


class ActorCritic(nn.Module):
    """Shared ActorCritic network architecture used by all PPO variants."""
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # Actor network
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

        # Critic network
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
    """Transition data structure used by all PPO variants."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class PPOTrainer:
    """Base PPO trainer with shared training logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def create_linear_schedule(self) -> Callable:
        """Create linear learning rate schedule."""
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.config["NUM_MINIBATCHES"] * self.config["UPDATE_EPOCHS"]))
                / self.config["NUM_UPDATES"]
            )
            return self.config["LR"] * frac
        return linear_schedule
    
    def compute_ppo_loss(
        self, 
        network: nn.Module,
        params: Any,
        traj_batch: Transition,
        advantages: jnp.ndarray,
        targets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple]:
        """Compute PPO loss (identical across all algorithms)."""
        # RERUN NETWORK
        pi, value = network.apply(params, traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.config["CLIP_EPS"],
                1.0 + self.config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.config["VF_COEF"] * value_loss
            - self.config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy, ratio)
    
    def calculate_gae(
        self,
        traj_batch: Transition,
        last_val: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate Generalized Advantage Estimation (identical across all algorithms)."""
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
            gae = (
                delta
                + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        targets = advantages + traj_batch.value
        return advantages, targets
    
    def update_minibatch(
        self,
        network: nn.Module,
        train_state: TrainState,
        batch_info: Tuple[Transition, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Update parameters using a minibatch (identical across all algorithms)."""
        traj_batch, advantages, targets = batch_info

        # Create loss function
        def _loss_fn(params):
            return self.compute_ppo_loss(network, params, traj_batch, advantages, targets)

        # Compute gradients and update
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        # Return loss info
        loss_info = {
            "total_loss": total_loss[0],
            "actor_loss": total_loss[1][1],
            "critic_loss": total_loss[1][0],
            "entropy": total_loss[1][2],
            "ratio": total_loss[1][3],
        }

        return train_state, loss_info
    
    def update_epoch(
        self,
        network: nn.Module,
        train_state: TrainState,
        traj_batch: Transition,
        advantages: jnp.ndarray,
        targets: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Run one epoch of updates (identical across all algorithms)."""
        # Shuffle minibatches
        batch_size = traj_batch.obs.shape[0]
        assert (
            batch_size == self.config["NUM_ACTORS"] * self.config["NUM_STEPS"]
        ), "batch size must be equal to number of actors * number of steps"
        permutation = jax.random.permutation(rng, batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.take(permutation, axis=0), batch
        )
        shuffled_batch, shuffled_advantages, shuffled_targets = batch

        # Process minibatches
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, [self.config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
            ),
            (shuffled_batch, shuffled_advantages, shuffled_targets),
        )

        # Update each minibatch
        def update_minibatch_wrapper(train_state, batch_info):
            return self.update_minibatch(network, train_state, batch_info)

        train_state, loss_info = jax.lax.scan(
            update_minibatch_wrapper, train_state, minibatches
        )
        
        return train_state, loss_info


# Utility functions for different batching strategies
def batchify(x: dict, agent_list, num_actors):
    """Batchify for IPPO/SPPPO (handles multiple agents)."""
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z, length):
        return jnp.concatenate(
            [z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1
        )

    x = jnp.stack(
        [x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list]
    )
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Unbatchify for IPPO/SPPPO."""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def batchify_main_agent(x: dict, main_agent: str, num_envs: int):
    """Batchify for FSPPPO (only main agent)."""
    return x[main_agent].reshape((num_envs, -1))


def get_main_agent_data(x: dict, main_agent: str):
    """Extract data for main agent only (FSPPPO)."""
    return x[main_agent]


def create_full_action_dict(
    main_action: jnp.ndarray,
    opponent_action: jnp.ndarray,
    main_agent: str,
    opponent_agent: str,
    num_envs: int,
):
    """Create action dictionary for both agents from separate action arrays (FSPPPO)."""
    return {
        main_agent: main_action.reshape((num_envs,)),
        opponent_agent: opponent_action.reshape((num_envs,)),
    }